# built-in
import os
import urllib.request

# 3rd

# ours
from mytools import fntime


"""
TODO Ubuntu Desctopの方で修正入れた気がする...要確認
"""
@fntime
def download_xml(year:str, path:str) -> None :
    dir_components = os.listdir(path)
    files = [f for f in dir_components if os.path.isfile(os.path.join(path, f))]
    file_count = 0
    for file_name in files :
        fr = open(f"{path}{file_name}", "r")
        print(file_count)
        file_count += 1
        for line in fr.readlines() :
            if line[69:73] != "00a9" : continue
            else:
                log_id = line[56:87]
                url = f"http://tenhou.net/0/log/?{log_id}"
                req = urllib.request.Request(url)
                save_path = f"../data/xml/{year}/{log_id}.xml"
                if os.path.isfile(save_path) : continue
                else : fw = open(save_path, "wb")
                with urllib.request.urlopen(req) as res:
                    body = res.read()
                    fw.write(body)
                    fw.close()
        fr.close()


if __name__ == "__main__" :
    year = "2019"
    path = f"../data/tenhou/{year}/"

    download_xml(year, path)

