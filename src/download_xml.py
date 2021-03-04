# built-in
import sys
import os
import urllib.request
from termcolor import colored

# 3rd

# ours
from mytools import fntime

# TODO 途中で止まってしまうのを修正
@fntime
def download_xml(year:str, path:str) -> None :
    # ../data/html/からhtmlファイルを読み込む
    # 1つのhtmlファイルに1日分のログIDが記録されている
    dir_components = os.listdir(path)
    files = [f for f in dir_components if os.path.isfile(os.path.join(path, f))]

    file_count = 0
    for file_name in files :
        fr = open(f"{path}{file_name}", "r")
        print(file_count)
        file_count += 1
        for line in fr.readlines() :
            # 鳳南のlogでなければスキップ
            if line[69:73] != "00a9" : continue
            else:
                # ログIDを抽出
                log_id = line[56:87]
                save_path = f"../data/xml/{year}/{log_id}.xml"

                # すでに取得済みのファイルはスキップ
                if os.path.isfile(save_path) : continue
                else : fw = open(save_path, "wb")

                # xmlを取得するリクエストを送る
                url = f"http://tenhou.net/0/log/?{log_id}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as res:
                    body = res.read()
                    fw.write(body)
                    fw.close()

        fr.close()

if __name__ == "__main__" :
    args = sys.argv
    if len(args) == 2 : year = args[1]
    else : print("Usage : " + colored("python download_xml.py {year}", "yellow", attrs=["bold"]))
    path = f"../data/html/{year}/"
    download_xml(year, path)
