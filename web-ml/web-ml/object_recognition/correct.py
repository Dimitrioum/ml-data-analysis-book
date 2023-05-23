# Корректировщик датасета предсказанных номеров автомобилей

import pandas as pd
import numpy as np
import datetime
import os


def correct_raw_data(azs, trk):

    print("\nResults are being corrected. Please, wait.")
    # Папка предсказанных номеров
    predicted_dir = "predictions" + os.sep + f"azs_{azs}_trk_{trk}" + os.sep

    def pandas_tabjoin(tablelist, sepr):
        df = pd.read_csv(tablelist[0], sep=sepr)
        # Итерирование и слияние таблиц
        for table in tablelist[1:]:
            df = df.append(pd.read_csv(table, sep=sepr))
        return df

    # Загрузка всех файлов с распознанными номерами
    tablelist = [predicted_dir + x for x in os.listdir(predicted_dir)]
    if len(tablelist) != 0:

        predicted = pandas_tabjoin(tablelist, sepr=",")

        # Создание колонки с длинной номеров
        predicted["plate_len"] = predicted["Plate"].apply(lambda x: len(x))

        # Фильтрация номеров по длине (допустимы 8 и 9 знаков в формате номеров РФ)
        predicted = predicted[
            (predicted["plate_len"] == 8) | (predicted["plate_len"] == 9)
        ]
        predicted.reset_index(inplace=True, drop=True)

        # Разбивка строки по символам
        predicted["plate_split"] = predicted["Plate"].apply(lambda x: [a for a in x])
        sep_chars = pd.DataFrame(predicted["plate_split"].values.tolist())

        # Обработка неоднозначных знаков распознавания
        for char, to_replace in zip(["B", "O", "T"], ["8", "0", "7"]):
            for i in [1, 2, 3, 6, 7, 8]:
                sep_chars[i] = np.where(sep_chars[i] == char, to_replace, sep_chars[i])

        for char, to_replace in zip(["8", "0", "7"], ["B", "O", "T"]):
            for i in [0, 4, 5]:
                sep_chars[i] = np.where(sep_chars[i] == char, to_replace, sep_chars[i])

        # Замена стандартной латиницы кириллицей
        for latin, cyr in zip(
            ["A", "B", "C", "E", "H", "K", "M", "O", "P", "T", "X", "Y"],
            ["А", "В", "С", "Е", "Н", "К", "М", "О", "Р", "Т", "Х", "У"],
        ):
            for i in [0, 4, 5]:
                sep_chars[i] = np.where(sep_chars[i] == latin, cyr, sep_chars[i])
        sep_chars.fillna("", inplace=True)

        temp = pd.DataFrame([])
        temp["index"] = sep_chars.index

        # Слияние знаков региона
        regions = sep_chars[[6, 7, 8]]
        sep_chars["region"] = temp["index"].apply(lambda x: "".join(regions.iloc[x]))

        # Слияние обработанных знаков
        temp = pd.DataFrame([])
        temp["index"] = sep_chars.index
        sep_chars_temp = sep_chars[[i for i in range(0, 9)]]
        sep_chars["Plate"] = temp["index"].apply(
            lambda x: "".join(sep_chars_temp.iloc[x])
        )

        # Проведене проверки логического расположения цифр и букв
        cyr = ["А", "В", "С", "Е", "Н", "К", "М", "О", "Р", "Т", "Х", "У"]
        digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ""]

        for col, cat, idx in zip(
            ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
            [cyr, digits, digits, digits, cyr, cyr, digits, digits, digits],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ):
            sep_chars[col] = sep_chars[idx].apply(lambda x: 1 if x in cat else 0)

        sep_chars["check"] = 0
        for col in ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]:
            sep_chars["check"] += sep_chars[col]

        predicted["Plate"] = sep_chars["Plate"]
        predicted["check"] = sep_chars["check"]
        predicted["region"] = sep_chars["region"]

        # Применение проверки логического расположения знаков
        predicted = predicted[predicted["check"] == 9].copy()

        # Возможные регионы автомобилей
        a = ["0" + str(x) for x in range(1, 10)]
        b = [str(x) for x in range(10, 100)]
        c = [
            "102",
            "113",
            "116",
            "121",
            "123",
            "124",
            "125",
            "126",
            "134",
            "136",
            "138",
            "142",
            "150",
            "152",
            "154",
            "159",
            "161",
            "163",
            "164",
            "173",
            "174",
            "177",
            "178",
            "186",
            "190",
            "196",
            "197",
            "198",
            "199",
            "277",
            "299",
            "716",
            "725",
            "750",
            "763",
            "777",
            "790",
            "799",
        ]
        # d = ['0' + str(x) for x in range(10, 100)]
        regions_dict = a + b + c

        predicted = predicted[predicted["region"].isin(regions_dict)]

        # Фильтрация распознанных знаков по вероятности 0.5
        p_filtered = predicted[predicted["Probability"] > 0.5].copy()
        p_filtered.reset_index(inplace=True, drop=True)

        # Удаление возможных дубликатов отфильтрованных номеров
        ###ВНИМАНИЕ### Это временная корректировка, с добавлением индексации по автомобилям лучше использовать дубликаты для частотной агрегации номеров
        # p_filtered.drop_duplicates(subset='Plate', inplace=True)

        # Простейшая частотная агрегация добавлена

        # Удаление технических полей проверки номеров
        df = p_filtered.drop(
            columns=["plate_len", "plate_split", "check", "region"], axis=1
        )

        # Отсев данных с битой нераспознанной алгоритмом датой
        try:
            total_rows = len(df)
            df = df[df["Time"] != "-"].copy()
            reduced_rows = len(df)
            print(
                "\nDropped %s of data without timestamps"
                % "{:.2%}".format(1 - (reduced_rows / total_rows))
            )
            print("\nTotal rows: %s\nUnique values:\n%s" % (total_rows, df.nunique()))
        except:
            pass

        # Корректировка датасета под шаблон для отправки в автокод
        ############################################################################

        def hamming_distance(s1, s2):
            return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

        def reconstruct(df):

            df.sort_values(by=["AZS", "Station", "Time"], inplace=True)
            df["date"] = pd.to_datetime(df["Time"]).dt.date
            df["detection_time"] = pd.to_datetime(
                df["Time"].apply(lambda x: str(x).split(".")[0])
            ).dt.time
            df.drop(columns=["Time", "index"], inplace=True)

            df.rename(
                columns={
                    "AZS": "azs",
                    "Station": "trk",
                    "Plate": "plate",
                    "Probability": "proba",
                },
                inplace=True,
            )

            df.reset_index(inplace=True, drop=True)

            ###################
            # Индексация схожих номеров по расстоянию Хеминга
            ###################
            c = 0
            df["sys"] = None
            for i in range(df.shape[0] - 1):
                if hamming_distance(df.iloc[i][2], df.iloc[i + 1][2]) <= 3:
                    df.loc[i, "sys"], df.loc[i + 1, "sys"] = c, c
                else:
                    c += 1
                    df.loc[i, "sys"], df.loc[i + 1, "sys"] = c - 1, c
            ###################

            df.sort_values(by=["sys"], inplace=True)
            df.reset_index(inplace=True, drop=True)

            ###################
            # Схлопывание повторяющихся номеров
            res_df = pd.DataFrame([])

            for sys in df["sys"].unique().tolist():

                tab = df[df["sys"] == sys].copy()
                tab.sort_values(by=["detection_time"], inplace=True)

                shape = tab.shape
                fix_start = tab.iloc[0]["detection_time"]
                fix_end = tab.iloc[shape[0] - 1]["detection_time"]
                max_proba = tab["proba"].max()

                tmp = pd.DataFrame(
                    [
                        df[(df["sys"] == sys) & (df["proba"] == max_proba)]
                        .iloc[0]
                        .to_dict()
                    ]
                )
                tmp["fix_start"] = fix_start
                tmp["fix_end"] = fix_end
                res_df = res_df.append(tmp)

            res_df.reset_index(inplace=True, drop=True)
            try:
                del res_df["sys"]
            except:
                pass
            return res_df

        df = reconstruct(df)
        # Сохранение датасета распознанных номеров
        str_part = str(datetime.datetime.now())[:10].replace("-", "_")
        path_name = os.path.join(
            "dataset", f"azs_{azs}_trk_{trk}", f"{str_part}_detections.csv"
        )

        df.to_csv(path_name, index=False)

    else:
        print("Not tables for making dataset.")
        pass
