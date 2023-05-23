
import os
import sqlalchemy as sa
import pandas as pd


class SQLiteUpdater:
    def __init__(self, db_dir, dataset_dir):
        self.db_dir = db_dir
        self.dataset_dir = dataset_dir

        # Чтение базы данных
        def dbase_connect(db_name):
            # 'sqlite:///dbsqlite.db'
            engine = sa.create_engine(db_name)
            connection = engine.connect()
            metadata = sa.MetaData()
            return engine, connection, metadata

        self.eng, self.conn, self.meta = dbase_connect(self.db_dir)
        # main_table
        self.main_table = sa.Table('detections_table',
                                   self.meta, autoload=True,
                                   autoload_with=self.eng)

    # Получение даты и времени последней детекции по базе данных
    def get_latest_db_date(self):
        # Проверка наличия данных
        query = sa.select([self.main_table])
        lst = self.conn.execute(query).fetchall()
        # Если общий запрос к базе дает пустой список, то вернуть int 0
        if len(lst) == 0:
            return 0
        # Иначе вернуть строку с максимальной датой
        else:
            results = self.conn.execute(sa.select([self.main_table])).fetchall()
            df = pd.DataFrame(results)
            df.columns = results[0].keys()
            return df['utc'].max()

    # Чтение таблиц в папке dataset, взятие данных > позднего времени по БД
    def read_dataset(self, datetime_threshold):
        # Формирование списка .csv-файлов
        folders = os.listdir(self.dataset_dir)
        folders = [os.path.join(self.dataset_dir, x) for x in folders]
        files = []
        for f in folders:
            tmp_files = os.listdir(f)
            for file in tmp_files:
                files.append(os.path.join(f, file))

        df = pd.DataFrame(columns=['azs', 'trk', 'date',
                                   'detection_time', 'plate',
                                   'proba', 'utc', 'fix_start',
                                   'fix_end', 'base64'])
        # Загрузка и фильтрация данных
        for f in files:
            print(f)
            tmp = pd.read_csv(f)
            df = df.append(tmp, sort=False)
            tmp = None
        df.reset_index(inplace=True, drop=True)
        if isinstance(datetime_threshold, int):
            var_pass = 0
            return df, var_pass
        else:
            df = df[df['utc'] > datetime_threshold]

            if df.shape[0] == 0:
               var_pass = 1
            else:
                var_pass = 0

            return df, var_pass

    # INSERT данных в SQLite
    def insert_data_to_sqldb(self, df):
        query = sa.insert(self.main_table)
        value_list = df.to_dict(orient='records')
        self.conn.execute(query, value_list)

    # Запуск обновления базы данных
    def run(self):
        #try:
        # Получение последенего timestamp-а
        th = self.get_latest_db_date()
        # Чтение датасета с учетом timestamp-а
        df, var_pass = self.read_dataset(th)

        if var_pass:
            pass
        else:
            # INSERT датасета
            self.insert_data_to_sqldb(df)
        #except:
            #pass
