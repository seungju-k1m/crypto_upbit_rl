from configuration import *
from Simulator.pipeline import DataPipeLine


def download_data():
    start_time = '2020-01-01 00:00:00'
    end_time = '2021-11-01 00:00:00'
    start_time_dt = datetime.fromisoformat(start_time)
    end_time_dt = datetime.fromisoformat(end_time)
    duration = timedelta(days=1)
    delta = int((end_time_dt - start_time_dt) / duration)
    for i in range(delta):
        start_time_dt = start_time_dt + duration
        tmp = start_time_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(tmp)
        x = DataPipeLine(
            to=tmp,
            duration=DURATION
        )
        

if __name__ == "__main__":
    download_data()