import asyncio
import logging
import signal
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand
from tg_bot.handlers import common, summarize, base_channels, ml_logic
from decouple import config


async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="/start", description="Запустить бота"),
        BotCommand(command="/summarize", description="Суммаризация по каналам"),
        BotCommand(command="/base_channels", description="Отслеживаемые каналы"),
        BotCommand(command="/help", description="Помощь"),
        BotCommand(command="/cancel", description="Отменить действие"),
        BotCommand(command="/about", description="Информация о проекте"),
    ]
    await bot.set_my_commands(commands)

bot = Bot(token=config('BOT_TOKEN'))


async def shutdown(dp: Dispatcher, bot: Bot):
    """
    Функция для завершения работы бота и диспетчера.
    """
    logging.info("Shutting down dispatcher and closing bot session...")
    await dp.fsm.storage.close()
    await dp.fsm.storage.wait_closed()
    await bot.session.close()


async def main(bot):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    dp = Dispatcher(storage=MemoryStorage())

    dp.include_routers(common.router,
                       summarize.router,
                       base_channels.router,
                       ml_logic.router)

    dp_task = asyncio.create_task(dp.start_polling(bot))

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def stop_signal_handler(*_):
        logging.info("Received stop signal, shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_signal_handler)

    await stop_event.wait()

    dp_task.cancel()
    try:
        await dp_task
    except asyncio.CancelledError:
        pass

    await shutdown(dp, bot)


if __name__ == '__main__':
    try:
        asyncio.run(main(bot))
    except KeyboardInterrupt:
        logging.info("Bot stopped manually")