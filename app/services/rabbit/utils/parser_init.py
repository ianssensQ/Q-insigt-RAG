from telethon import TelegramClient
import asyncio
from decouple import config

api_id = config('API_ID', cast=int)
api_hash = config('API_HASH')
phone_number = config('PHONE_NUMBER')
confirmation_code = config('CONFIRMATION_CODE', default=None)

client = TelegramClient("Parser_session", api_id, api_hash)

async def start():
    try:
        await client.connect()
        if not await client.is_user_authorized():
            print("Пользователь не авторизован. Попытка авторизации...")
            print(confirmation_code)
            if confirmation_code:
                await client.send_code_request(phone_number)
                await client.sign_in(phone_number, confirmation_code)
            else:
                await client.start(phone=lambda: phone_number)

        print("Пользователь успешно авторизован.")
        
        me = await client.get_me()
        print(f"Вы вошли как: {me.first_name} (@{me.username})")

    except Exception as e:
        print("Произошла ошибка:", e)
    finally:
        await client.disconnect()

if __name__ == '__main__':
    asyncio.run(start())
