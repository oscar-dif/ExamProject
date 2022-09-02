from flask import *
import datetime
import psycopg2
import pandas as pd
from forecast_model import forecast_price

df_flask ={}
def read_db(C):
    cur = C.cursor()
    cur.execute('''SELECT overallqual AS "overall quality" , 
                AVG(price_per_sqft) AS "average price per square feet" 
                FROM houses
                GROUP BY overallqual
                ORDER BY overallqual desc;''')
    rows = cur.fetchall()
    cur.close()
    return  rows
#dum= read_phonelist(conn)

def insert_entry(yearbuilt,
                    yearremoddad,
                    overallqual, 
                    exterqual,
                    garagecars,
                    kitchenqual,
                    bsmtqual):
    X = {
    'yearbuilt': yearbuilt,
    'yearremodadd': yearremoddad,
    'overallqual': overallqual,
    'exterqual': exterqual,
    'garagecars': garagecars,
    'kitchenqual':  kitchenqual,
    'bsmtqual': bsmtqual
        }
    X = pd.DataFrame(X, index=[0])

    return X

  
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
            database="houses_db",
            user="houses_user",
            password="123"
        )
    dum= read_db(conn)
    now = datetime.datetime.now()
    D = [str(now.year%100), str(now.month), str(now.day)]
    if len(D[1])<2:
        D[1] = '0'+D[1]
    if len(D[2])<2:
        D[2] = '0'+D[2]
    return render_template('list.html', list=dum, date=D)

@app.route("/insert")
def insert_function():
    yearbuilt = request.args['yearbuilt']
    yearremodadd = request.args['yearremodadd']
    overallqual = request.args['overallqual']
    exterqual = request.args['exterqual']
    garagecars = request.args['garagecars']
    kitchenqual = request.args['kitchenqual']
    bsmtqual = request.args['bsmtqual']

    X =insert_entry(int(yearbuilt),
                    int(yearremodadd),
                    int(overallqual), 
                    int(exterqual),
                    int(garagecars),
                    int(kitchenqual),
                    int(bsmtqual))
    prediction= forecast_price(X)
    return render_template('insert.html', nparray_flask=prediction)

if __name__ == "__main__":
    app.run(host='localhost')