from AI import BlockAI
from datetime import datetime
import time
from pprint import pprint
from Finance import Wallet

BLK = BlockAI('DEFAULT')
pocket = BLK.NowPredictions_ALL()
# inicialize
Wallets = {}
for i in pocket[2].keys():
    Wallets[i] = Wallet()
    Wallets[i].deposit(100)
# inicialize

while True:
    h = datetime.now().hour
    m = datetime.now().minute
    s = datetime.now().second
    print(f'sleep: {23 - h} hours {55 - m} minutes {60 - s} seconds')
    time.sleep(abs(60 * 60 * (23 - h) + 60 * (55 - m) + (60 - s)))

    BLK = BlockAI('DEFAULT')
    pocket = BLK.NowPredictions_ALL()
    BLK.SET_predictions(pocket)

    for i in range(2):
        with open('TRADING', 'a') as f:
            f.write(f'---------------------------------{datetime.now()}---------------------------------\n')
            for i in pocket[2].keys():
                try:
                    if pocket[0][i] >= pocket[2][i] * 1.01:
                        Wallets[i].buy_coin_usd(i, Wallets[i]['USDT'])
                    if pocket[0][i] <= pocket[2][i] * 0.99:
                        Wallets[i].sell_coin_quantity(i, Wallets[i][i])
                    ALL_Money = Wallets[i].ALL_IN_USD()
                    print(f'{i}: {ALL_Money}')
                    print(f'{i}: {Wallets[i]}')
                    f.write(f'{i}: {ALL_Money}\n')
                    f.write(f'{i}: {Wallets[i]}\n')
                except:
                    pass
            f.write(f'---------------------------------{datetime.now()}---------------------------------\n\n\n')

