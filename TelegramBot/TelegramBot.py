import Config
from AI import BlockAI
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

################################################################################################################
# Основа
bot = Bot(token=Config.BotToken)
dp = Dispatcher(bot)
################################################################################################################
# клавиатура
btn1 = KeyboardButton("/Today_predictions")
btn2 = KeyboardButton("/Express")
mainMenu = ReplyKeyboardMarkup(resize_keyboard=True).add(btn1).add(btn2)


################################################################################################################


@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await message.reply('HI', reply_markup=mainMenu)


@dp.message_handler(commands=['e'])
async def predictions_command(msg: types.Message):
    tests = BLK.Testing_Netwoks()
    r1 = []
    r2 = []
    for i in tests[0].keys():
        r1.append(f'{i} : {tests[0][i]}')
        r2.append(f'{i} : {tests[1][i]}')

    await bot.send_message(msg.from_user.id, str('\n'.join(r1)))
    await bot.send_message(msg.from_user.id, str('\n'.join(r2)))

@dp.message_handler(commands=['Express'])
async def predictions_command(msg: types.Message):
    pack = BLK.NowPredictions_ALL()
    BLK.SET_predictions(pack)
    res = BLK.DEL_recently_predictions()
    await bot.send_message(msg.from_user.id, str(res))

@dp.message_handler(commands=['Today_predictions'])
async def predictions_command(msg: types.Message):
    predictions = BLK.GET_today_predictions()
    await bot.send_message(msg.from_user.id, predictions)


if __name__ == '__main__':
    BLK = BlockAI('DEFAULT')
    executor.start_polling(dp)
