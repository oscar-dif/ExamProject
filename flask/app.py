from flask import *
import datetime
import psycopg2

#conn = sqlite3.connect("phone.db")

#dum = [
# ['arne', '013-131313'], ['berith','01234'], ['caesar','077-1212321']
#]
#dum=conn
#conn = psycopg2.connect(
#            host="localhost",
#            database="phone",
#            user="phone",
#            password="de2022"
#        )


def read_phonelist(C):
    cur = C.cursor()
    cur.execute("SELECT name, phone FROM phonelist;")
    rows = cur.fetchall()
    cur.close()
    return rows
#dum= read_phonelist(conn)

def insert_phone_entry(c, name, phone):
    cur = c.cursor()
    cur.execute(f"INSERT INTO phonelist VALUES ('{name}', '{phone}');")
    cur.execute("COMMIT;")
    cur.close()
    
def delete_phone(c, name):
    cur = c.cursor()
    cur.execute(f"DELETE FROM phonelist WHERE name = '{name}';")
    cur.execute("COMMIT;")
    cur.close()


app = Flask(__name__) #set main (__name__=main)

@app.route("/")
def start():
    conn = psycopg2.connect(
            host="localhost",
            database="phone",
            user="phone",
            password="de2022"
        )
    dum= read_phonelist(conn)
    now = datetime.datetime.now()
    D = [str(now.year%100), str(now.month), str(now.day)]
    if len(D[1])<2:
        D[1] = '0'+D[1]
    if len(D[2])<2:
        D[2] = '0'+D[2]
    return render_template('list.html', list=dum, date=D)

@app.route("/delete")
def delete_function():
    print('delete executed')
    conn = psycopg2.connect(
            host="localhost",
            database="phone",
            user="phone",
            password="de2022"
        )
    name = request.args['name']
    delete_phone(conn, name)
    return render_template('delete.html', name=name)

@app.route("/insert")
def insert_function():
    conn = psycopg2.connect(
            host="localhost",
            database="phone",
            user="phone",
            password="de2022"
        )
    name = request.args['name']
    phone = request.args['phone']
    insert_phone_entry(conn, name, phone)
    return render_template('insert.html', name=name, phone=phone)

if __name__ == "__main__":
    app.run(host='localhost')