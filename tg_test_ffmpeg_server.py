import logging
import asyncio
import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, ApplicationBuilder, ExtBot

from net_stream.bilibli_live import BilibiliLive
from net_stream.ffmpeg_server import FFmpegServer
from silero_vad import load_silero_vad
from translate.llm_translate import OpenAICompatibleLLMProvider
from transcribe.provider.faster_whisper import FasterWhisperBlockTranscriber
from audio_buffer import AudioBuffer

# --- Configuration ---
TELEGRAM_BOT_TOKEN = "..."
FFMPEG_PATH = "..." # Or None if ffmpeg is in PATH
TRANSLATE_API_KEY = "..."
TRANSLATE_BASE_URL = "..."
TRANSLATE_MODEL = "..."

VAD_THRESHOLD = 0.25
VAD_CUT_OFF_SAMPLES = 38000
MIN_SPEECH_SAMPLES = 4000

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx verbosity
logger = logging.getLogger(__name__)

# --- Global State ---
target_chat_id: int | None = None
target_room_id: int | None = None
translation_task: asyncio.Task | None = None


async def translate_worker(
    queue: asyncio.Queue[str | None],
    bot: ExtBot,
    chat_id: int,
    llm_provider: OpenAICompatibleLLMProvider
):
    """Worker task to fetch text from queue, translate, and send to Telegram."""
    logger.info(f"Translate worker started for chat {chat_id}")
    while True:
        try:
            src_text = await queue.get()
            if src_text is None:
                logger.info(f"Translate worker for chat {chat_id} received stop signal.")
                queue.task_done()
                break

            if not src_text.strip():
                logger.info("Skipping empty source text.")
                queue.task_done()
                continue

            logger.info(f"Translating for chat {chat_id}: {src_text[:50]}...")
            result = await llm_provider.translate(src_text)
            logger.info(f"Translation result for chat {chat_id}: {result[:50]}...")

            message_text = f"{src_text}\n---\n{result}"
            if len(message_text) > 4096:
                 logger.warning(f"Message too long ({len(message_text)} chars), sending truncated.")
                 message_text = message_text[:4090] + "\n[...]"

            await bot.send_message(chat_id=chat_id, text=message_text)

            queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"Translate worker for chat {chat_id} cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in translate worker for chat {chat_id}: {e}", exc_info=True)
            try:

                await bot.send_message(chat_id=chat_id, text=f"An error occurred during translation: {e}")
            except Exception as send_e:
                logger.error(f"Failed to send error message to chat {chat_id}: {send_e}")
            if 'src_text' in locals() and src_text is not None:
                 try:
                     queue.task_done()
                 except ValueError:
                     pass
    logger.info(f"Translate worker finished for chat {chat_id}")


async def run_live_translation(bot: ExtBot, chat_id: int, room_id: int):
    global target_chat_id, target_room_id, translation_task

    translate_queue: asyncio.Queue[str | None] = asyncio.Queue()
    livestream = None
    translate_task = None

    try:
        logger.info(f"Starting live translation process for chat {chat_id}, room {room_id}")

        llm_provider = OpenAICompatibleLLMProvider(
            base_url=TRANSLATE_BASE_URL,
            api_key=TRANSLATE_API_KEY,
            model=TRANSLATE_MODEL
        )
        translate_task = asyncio.create_task(
            translate_worker(translate_queue, bot, chat_id, llm_provider)
        )

        logger.info("Loading VAD model...")
        vad_model = load_silero_vad()
        vad_config = {
            "threshold": VAD_THRESHOLD,
            "cut_off_samples": VAD_CUT_OFF_SAMPLES,
        }
        logger.info("VAD model loaded.")

        logger.info("Initializing transcriber...")
        faster_whisper_stt = FasterWhisperBlockTranscriber(
            {"model_size_or_path": "large-v2", "download_root": "./whisper_cache/"}
        )
        logger.info("Transcriber initialized.")

        logger.info(f"Connecting to Bilibili room {room_id}...")
        #livestream = BilibiliLive(room_id=room_id)
        livestream = FFmpegServer(bind_ip="127.0.0.1", bind_port=6667)
        await livestream.spin_ffmpeg(ffmpeg_path=FFMPEG_PATH)
        logger.info(f"Connected to Bilibili room {room_id} and ffmpeg started.")

        audio_buffer = AudioBuffer()
        cont_non_speech = 0

        await bot.send_message(chat_id=chat_id, text=f"✅ Live translation started for room {room_id}!")

        while True:
            audio = await livestream.read_audio()
            if audio is None:
                logger.warning(f"Received None from Bilibili audio stream (room {room_id}), ending loop.")
                await bot.send_message(chat_id=chat_id, text=f"Stream from room {room_id} seems to have ended.")
                break

            audio_buffer.submit(audio)
            audio_tensor = torch.from_numpy(audio.astype('float32'))
            speech_prob = vad_model(audio_tensor, 16000).item()

            if speech_prob < vad_config["threshold"]:
                cont_non_speech += len(audio)
            else:
                cont_non_speech = 0

            if cont_non_speech > vad_config["cut_off_samples"]:
                speech_samples = audio_buffer.n_samples() - cont_non_speech
                if speech_samples >= MIN_SPEECH_SAMPLES:
                    speech_audio_np = audio_buffer.as_nparray()[:-cont_non_speech // 2]
                    logger.info(f"Transcribing {speech_audio_np.shape[0] / 16000:.2f}s of audio from room {room_id}...")
                    try:
                        transcript = faster_whisper_stt.transcribe(
                            speech_audio_np,
                            "",
                            segment_max_no_speech_prob=0.75,
                            segments_merge_fn=lambda x: " ".join(x),
                            language=None
                        )
                        logger.info(f"Transcript (Room {room_id}): '{transcript}'")
                        if transcript and transcript.strip():
                            await translate_queue.put(transcript.strip())
                        else:
                             logger.warning(f"Empty transcript received from room {room_id}, skipping.")
                    except Exception as e:
                        logger.error(f"Transcription error (Room {room_id}): {e}", exc_info=True)
                        try:
                            await bot.send_message(chat_id=chat_id, text=f"Transcription error for room {room_id}: {e}")
                        except Exception as send_e:
                            logger.error(f"Failed to send transcription error message to chat {chat_id}: {send_e}")

                audio_buffer.trim_head(audio_buffer.n_samples() - cont_non_speech // 2)
                cont_non_speech = audio_buffer.n_samples()

    except asyncio.CancelledError:
        logger.info(f"Live translation task cancelled for chat {chat_id}, room {room_id}.")
        try:
            await bot.send_message(chat_id=chat_id, text=f"⏹️ Live translation stopped for room {room_id}.")
        except Exception as send_e:
            logger.error(f"Failed to send stop confirmation to chat {chat_id}: {send_e}")
    except Exception as e:
        logger.error(f"Error in live translation task for chat {chat_id}, room {room_id}: {e}", exc_info=True)
        try:
            await bot.send_message(chat_id=chat_id, text=f"❌ An error occurred in the live translation process for room {room_id}: {e}")
        except Exception as send_e:
             logger.error(f"Failed to send error message to chat {chat_id}: {send_e}")
    finally:
        logger.info(f"Cleaning up resources for chat {chat_id}, room {room_id}...")
        if 'translate_queue' in locals() and translate_task and not translate_task.done():
            try:
                await translate_queue.put(None)
            except Exception as qe:
                 logger.error(f"Error putting None sentinel in queue for chat {chat_id}: {qe}")

        if translate_task:
            try:
                logger.info(f"Waiting for translate worker to finish for chat {chat_id}...")
                await asyncio.wait_for(translate_task, timeout=10.0)
                logger.info(f"Translate worker finished gracefully for chat {chat_id}.")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for translate worker (chat {chat_id}). Cancelling it.")
                translate_task.cancel()
            except asyncio.CancelledError:
                 logger.info(f"Translate worker (chat {chat_id}) was already cancelled.")
            except Exception as e:
                 logger.error(f"Error waiting for translate worker (chat {chat_id}): {e}")

        if livestream:
            logger.info(f"Closing Bilibili connection for room {room_id}...")
            try:
                if hasattr(livestream, 'close') and asyncio.iscoroutinefunction(livestream.close):
                     await livestream.close()
                elif hasattr(livestream, 'close'):
                     livestream.close()
                elif livestream.ffmpeg_process and livestream.ffmpeg_process.returncode is None:
                     logger.info(f"Terminating ffmpeg process for room {room_id}...")
                     livestream.ffmpeg_process.terminate()
                     try:
                         await asyncio.wait_for(livestream.ffmpeg_process.wait(), timeout=5.0)
                         logger.info(f"ffmpeg process terminated for room {room_id}.")
                     except asyncio.TimeoutError:
                         logger.warning(f"ffmpeg (room {room_id}) did not terminate gracefully, killing.")
                         livestream.ffmpeg_process.kill()
                     except Exception as e:
                          logger.error(f"Error waiting for ffmpeg termination (room {room_id}): {e}")
                else:
                     logger.info(f"ffmpeg process already terminated or not found for room {room_id}.")
            except Exception as e:
                logger.error(f"Error closing Bilibili connection (room {room_id}): {e}", exc_info=True)

        current_task = asyncio.current_task()
        if target_chat_id == chat_id and translation_task is current_task:
             logger.info(f"Resetting global state as task for chat {chat_id}, room {room_id} ends.")
             target_chat_id = None
             target_room_id = None
             translation_task = None
        else:
             logger.warning(f"Task for chat {chat_id}, room {room_id} ended, but global state indicates different target ({target_chat_id}, {target_room_id}) or task ({translation_task}). State not reset by this task.")


# --- Telegram Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command with a room_id argument."""
    global target_chat_id, target_room_id, translation_task
    current_chat_id = update.effective_chat.id
    user = update.effective_user

    logger.info(f"Received /start command from user {user.id} in chat {current_chat_id} with args: {context.args}")

    if not context.args:
        # Removed parse_mode, removed backticks
        await update.message.reply_text(
            "Please provide a Bilibili Room ID after /start.\n"
            "Example: /start 123456"
        )
        return

    try:
        room_id_to_start = int(context.args[0])
        if room_id_to_start <= 0:
             raise ValueError("Room ID must be positive.")
    except (ValueError, IndexError):
        await update.message.reply_text(
            "Invalid Room ID. Please provide a valid positive number.\n"
            "Example: /start 123456"
        )
        return

    if target_chat_id is None:
        target_chat_id = current_chat_id
        target_room_id = room_id_to_start
        await update.message.reply_text(
            f"Hi {user.first_name}! Received /start command for room {target_room_id}.\n"
            f"Starting live translation stream to this chat ({target_chat_id}).\n"
            "Use /stop to end the translation."
        )
        logger.info(f"Set target chat ID to {target_chat_id} and room ID to {target_room_id}. Starting background task.")

        translation_task = asyncio.create_task(
            run_live_translation(context.bot, target_chat_id, target_room_id)
        )
        translation_task.add_done_callback(handle_task_completion)

    elif current_chat_id == target_chat_id:
        logger.info(f"Ignoring /start from target chat {current_chat_id}, already running for room {target_room_id}.")
        await update.message.reply_text(
             f"Live translation is already running for room {target_room_id} in this chat. Use /stop to end it."
        )
    else:
        logger.info(f"Ignoring /start from chat {current_chat_id}, currently serving chat {target_chat_id} (room {target_room_id}).")
        await update.message.reply_text(
            f"Sorry, the bot is currently busy translating for room {target_room_id} in another chat. Please try again later."
        )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /stop command."""
    global target_chat_id, target_room_id, translation_task
    current_chat_id = update.effective_chat.id
    user = update.effective_user

    logger.info(f"Received /stop command from user {user.id} in chat {current_chat_id}")

    if target_chat_id is None:
        await update.message.reply_text("No translation is currently running.")
    elif current_chat_id == target_chat_id:
        if translation_task and not translation_task.done():
            logger.info(f"Stopping translation task for chat {current_chat_id}, room {target_room_id}.")
            await update.message.reply_text(f"Stopping live translation for room {target_room_id}...")
            translation_task.cancel()
        else:
            logger.warning(f"Stop command received for chat {current_chat_id}, but task is already done or None.")
            await update.message.reply_text("Translation seems to have already stopped.")
            target_chat_id = None
            target_room_id = None
            translation_task = None
    else:
        logger.info(f"Ignoring /stop from chat {current_chat_id}, currently serving chat {target_chat_id} (room {target_room_id}).")
        await update.message.reply_text(
            f"You can only stop the translation from the chat where it was started (currently chat ID {target_chat_id})."
        )


def handle_task_completion(task: asyncio.Task) -> None:
    """Callback function to handle when the background task finishes or fails."""
    global target_chat_id, target_room_id, translation_task
    try:
        exception = task.exception()
        if exception:
            logger.error(f"Translation task failed with exception: {exception}", exc_info=exception)
    except asyncio.CancelledError:
        logger.info("Translation task was cancelled (detected in completion handler).")
    finally:
        if translation_task is task:
             logger.info("Resetting global state via task completion handler.")
             target_chat_id = None
             target_room_id = None
             translation_task = None
        else:
             logger.warning("A task completed, but it wasn't the currently tracked translation_task. State not reset by handler.")


# --- Main Bot Execution ---

def main() -> None:
    """Starts the bot."""
    logger.info("Starting bot...")
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))

    logger.info("Bot is polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot stopped.")


if __name__ == "__main__":
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set!")
        exit(1)
    if not TRANSLATE_API_KEY or "..." in TRANSLATE_API_KEY:
        logger.warning("DEEPSEEK_API_KEY is not set or is a placeholder!")


    main()
