import sqlite3 as sql


class DBConnector:
    def __init__(self, database, reset=True, same_thread=True):
        self.connection = sql.connect(database, check_same_thread=same_thread)
        if reset:
            self.init_tables()

    def __del__(self):
        """ Destructor closes the connection to the database"""
        if self.connection:
            self.connection.close()

    def execute_query(self, cmd, attr=()):
        cursor = self.connection.cursor()
        cursor.execute(cmd, attr)
        data = cursor.fetchall()
        cursor.close()
        return data

    def execute_non_query(self, cmd, attr=()):
        cursor = self.connection.cursor()
        cursor.execute(cmd, attr)
        cursor.close()
        self.connection.commit()

    def insert_many(self, table_name, data):
        cursor = self.connection.cursor()
        cursor.executemany(
            'INSERT INTO {0} VALUES ({1})'.format(table_name, ''.join(['?,' for i in range(len(data[0]))])[:-1]), data)
        cursor.close()

        self.connection.commit()

    def init_tables(self):
        cursor = self.connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS counter")
        cursor.execute("DROP TABLE IF EXISTS calibration")
        cursor.execute("DROP TABLE IF EXISTS dht11")
        cursor.execute("DROP TABLE IF EXISTS bmp180")
        cursor.execute(""" CREATE TABLE IF NOT EXISTS counter ( gate_id TEXT, count_in INTEGER, count_out INTEGER,
                                                                time INTEGER) """)
        cursor.execute(""" CREATE TABLE IF NOT EXISTS calibration ( gate_id TEXT, in_cali REAL, out_cali REAL) """)
        cursor.execute(""" CREATE TABLE IF NOT EXISTS dht11 ( time INTEGER, temperature REAL, humidity REAL) """)
        cursor.execute(""" CREATE TABLE IF NOT EXISTS bmp180 ( time INTEGER, temperature REAL, pressure REAL) """)
        self.connection.commit()

    def get_calibration_data(self):
        return self.execute_query("SELECT gate_id, in_cali, out_cali FROM calibration")
