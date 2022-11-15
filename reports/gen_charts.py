import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

dffootprint = pd.read_csv('./memfootprint.csv')
dfspenttime = pd.read_csv('./spenttime.csv')

sources = ['zfp_rate8_4gpu_10iter-each.json', 'zfp_rate4_4gpu_10iter-each.json', 'cusz_abs_4gpu_10iter-each.json', 'cusz_r2r_4gpu_10iter-each.json']
devices = 4

def memfootprint_chart(source, device):
    df = dffootprint[dffootprint['MemKind'] == 2]
    df = df[df['DeviceId'] == device]
    df = df[df['Source'] == source]
    df = df.sort_values('t_sec')

    plt.plot(df.t_sec, df.MegaBytes.cumsum())

    source = source.replace('.json', '')
    plt.savefig(f'./charts/memoryfootprint_{source}_device_{device}.png')
    plt.clf()
    print('sum memfootprint for', source, 'device#', device, ':', df.Bytes.sum(), 'bytes')

def spent_time_report(source):
    df = dfspenttime[dfspenttime['Source'] == source]

    if 'zfp' in source:
        print(source, ': ZFP_COMPRESS spent time mean:', df[df['EventName'] == 'ZFP_COMPRESS'].SpentTime.mean() / 1000, 'us')
        print(source, ': ZFP_COMPRESS spent time std:', df[df['EventName'] == 'ZFP_COMPRESS'].SpentTime.std() / 1000, 'us')
        print(source, ': ZFP_COMPRESS spent time median:', df[df['EventName'] == 'ZFP_COMPRESS'].SpentTime.median() / 1000, 'us')
        print(source, ': ZFP_DECOMPRESS spent time mean:', df[df['EventName'] == 'ZFP_DECOMPRESS'].SpentTime.mean()/ 1000, 'us')
        print(source, ': ZFP_DECOMPRESS spent time std:', df[df['EventName'] == 'ZFP_DECOMPRESS'].SpentTime.std()/ 1000, 'us')
        print(source, ': ZFP_DECOMPRESS spent time median:', df[df['EventName'] == 'ZFP_DECOMPRESS'].SpentTime.median()/ 1000, 'us')
    else:
        print(source, ': CUSZ_COMPRESS spent time mean:', df[df['EventName'] == 'CUSZ_COMPRESS'].SpentTime.mean()/ 1000, 'us')
        print(source, ': CUSZ_COMPRESS spent time std:', df[df['EventName'] == 'CUSZ_COMPRESS'].SpentTime.std()/ 1000, 'us')
        print(source, ': CUSZ_COMPRESS spent time median:', df[df['EventName'] == 'CUSZ_COMPRESS'].SpentTime.median()/ 1000, 'us')
        print(source, ': CUSZ_DECOMPRESS spent time mean:', df[df['EventName'] == 'CUSZ_DECOMPRESS'].SpentTime.mean()/ 1000, 'us')
        print(source, ': CUSZ_DECOMPRESS spent time std:', df[df['EventName'] == 'CUSZ_DECOMPRESS'].SpentTime.std()/ 1000, 'us')
        print(source, ': CUSZ_DECOMPRESS spent time median:', df[df['EventName'] == 'CUSZ_DECOMPRESS'].SpentTime.median()/ 1000, 'us')


for source in sources:
    for device in range(devices):
        memfootprint_chart(source, device)

print('----------')

for source in sources:
    spent_time_report(source)


