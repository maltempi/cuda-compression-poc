import json
import pandas as pd

filepaths = [
    'zfp_rate8_4gpu_10iter-each/zfp_rate8_4gpu_10iter-each.json',
    'zfp_rate4_4gpu_10iter-each/zfp_rate4_4gpu_10iter-each.json',
    'cusz_r2r_4gpu_10iter-each/cusz_r2r_4gpu_10iter-each.json',
    'cusz_abs_4gpu_10iter-each/cusz_abs_4gpu_10iter-each.json',
]

spenttime = {
    "EventName": [],
    "Source": [],
    "GlobalTid": [],
    "Timestamp": [],
    "EndTimestamp": [],
    "SpentTime": [],
}

memfootprint = {
    "Source": [],
    "DeviceId": [],
    "t_nano": [],
    "t_sec": [],
    "MemKind": [],
    "Bytes": [],
    "MegaBytes": [],
    "OpType": [],
}

for filepath in filepaths:
    with open(filepath) as f:
        while True:
            line = f.readline()
            if not line:
                break

            entry = json.loads(line)
            
            if 'CudaMemoryUsageEvent' in entry:
                memUsage = entry['CudaMemoryUsageEvent']
                memfootprint['Source'].append(filepath.split('/')[-1])
                memfootprint['DeviceId'].append(str(memUsage['deviceId']))
                memfootprint['t_nano'].append(int(memUsage['startNs']))
                memfootprint['t_sec'].append(int(memUsage['startNs']) * 1e-9)
                memfootprint['MemKind'].append(memUsage['memKind'])
                if memUsage['type'] == 'deallocation':
                    memfootprint['Bytes'].append(int(memUsage['bytes']) * -1)
                    memfootprint['MegaBytes'].append((int(memUsage['bytes']) / 1024 / 1024) * -1)
                    
                else:
                    memfootprint['Bytes'].append(int(memUsage['bytes']))
                    memfootprint['MegaBytes'].append(int(memUsage['bytes']) / 1024 / 1024)
                
                memfootprint['OpType'].append(memUsage['type'])
                continue

            if 'NvtxEvent' in entry:
                nvtxEvent = entry['NvtxEvent']

                if nvtxEvent['Text'] in ['CUSZ_DECOMPRESS', 'CUSZ_COMPRESS', 'ZFP_DECOMPRESS', 'ZFP_COMPRESS']:
                    spenttime['EventName'].append(nvtxEvent['Text'])
                    spenttime['GlobalTid'].append('_' + str(nvtxEvent['GlobalTid']))
                    spenttime['Timestamp'].append(int(nvtxEvent['Timestamp']))
                    spenttime['EndTimestamp'].append(int(nvtxEvent['EndTimestamp']))
                    spenttime['SpentTime'].append(int(nvtxEvent['EndTimestamp']) - int(nvtxEvent['Timestamp']))
                    spenttime['Source'].append(filepath.split('/')[-1])
                continue



df = pd.DataFrame(spenttime)
df.to_csv('spenttime.csv')

df = pd.DataFrame(memfootprint)
df.to_csv('memfootprint.csv')

