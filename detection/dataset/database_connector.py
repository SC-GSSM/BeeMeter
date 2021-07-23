import sqlite3 as sql


class Database:
    def __init__(self, path):
        self.connection = sql.connect(path)
        self.create_tables()

    def __del__(self):
        """ Destructor closes the connection to the database"""
        if self.connection:
            self.connection.close()

    def len(self):
        return "Images: {0}, Labels: {1}".format(self.number_image(), self.number_label())

    def number_image(self):
        stmt = """SELECT COUNT(*) FROM image"""
        return self.reduce_list(self.send_cmd(stmt))[0]

    def number_label(self):
        stmt = """SELECT COUNT(*) FROM label"""
        return self.reduce_list(self.send_cmd(stmt))[0]

    def send_cmd(self, cmd, attr=()):
        cursor = self.connection.cursor()
        cursor.execute(cmd, attr)
        data = cursor.fetchall()
        cursor.close()
        return data

    def reduce_list(self, li):
        """ Gets a list of tuples and return a 1d list """
        return [item for t in li for item in t]

    def create_tables(self):
        tbl_img = """CREATE TABLE image(ID INTEGER PRIMARY KEY AUTOINCREMENT, filename text);"""
        tbl_lbl = """ CREATE TABLE label(ID INTEGER PRIMARY KEY AUTOINCREMENT,
                                         image_ID INTEGER,
                                         x1 INTEGER,
                                         y1 INTEGER,
                                         x2 INTEGER,
                                         y2 INTEGER);"""
        tbl_tbl = "SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%';"
        existing_tbl = self.reduce_list(self.send_cmd(tbl_tbl))

        if "image" not in existing_tbl:
            self.send_cmd(tbl_img)
        if "label" not in existing_tbl:
            self.send_cmd(tbl_lbl)

    def reset_db(self):
        self.send_cmd("drop table image")
        self.send_cmd("drop table label")
        self.connection.commit()

    def image_id_by_filename(self, filename):
        """
        Get the ID of an image in the database using the filename.
        :param filename: The name of the image
        :return: tuple whose first value indicates whether the operation gets a result and the second value is the
        result
        """
        stmt = 'SELECT ID FROM image where filename = ?'
        img_id = self.send_cmd(stmt, attr=(filename,))
        if img_id:
            return True, img_id[0][0]
        else:
            return False, 0

    def label_coords_by_image_id(self, image_id):
        """

        :param image_id:
        :return:
        """
        stmt = 'SELECT X1, Y1, X2, Y2 FROM label WHERE image_id=?'
        labels = self.send_cmd(stmt, attr=(image_id,))
        return labels

    def add_image(self, filename):
        stmt = """ INSERT INTO image(filename) VALUES (?)"""
        self.send_cmd(stmt, (filename,))
        self.connection.commit()
        return self.send_cmd("select last_insert_rowid();")

    def add_label(self, image_ID, x1, y1, x2, y2):
        stmt = """ INSERT INTO label (image_ID, x1, y1, x2, y2) VALUES (?,?,?,?,?)"""
        self.send_cmd(stmt, (image_ID, int(x1), int(y1), int(x2), int(y2)))
        self.connection.commit()

    def delete_labels_by_image_id(self, image_ID):
        stmt = """ DELETE FROM label WHERE image_ID = (?)"""
        self.send_cmd(stmt, (image_ID,))
        self.connection.commit()
