from flask import Flask, render_template, json, request
from flaskext.mysql import MySQL
from werkzeug import generate_password_hash, check_password_hash
from flask import session, redirect
from market import *
from market_ml import *

mysql = MySQL()
app = Flask(__name__)
app.secret_key = 'why would I tell you my secret key?'

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'optimize'
app.config['MYSQL_DATABASE_DB'] = 'StockMarketApp'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)


@app.route("/")
def main():
    return render_template('index.html')


@app.route('/showSignUp')
def showSignUp():
    return render_template('signup.html')

'''
@app.route('/userHome/analysis', methods=['ANALYZE'])
def analyze_ticker():
    try:
        ticker = request.form['inputTicker']
        print("Button clicked for " + ticker)
        analysis = 'AAPL is overvalued by 3 percent' #get_analysis_text(ticker)
        print(analysis)
        return render_template('userHome.html', report=analysis)
    except Exception as e:
        print('failed')
        return json.dumps({'error':str(e)})
'''

@app.route('/signUp',methods=['POST'])
def signUp():
    try:
        _name = request.form['inputName']
        _email = request.form['inputEmail']
        _password = request.form['inputPassword']

        # validate the received values
        if _name and _email and _password:
            
            # All Good, let's call MySQL
            
            conn = mysql.connect()
            cursor = conn.cursor()
            _hashed_password = generate_password_hash(_password)
            cursor.callproc('sp_createUser',(_name,_email,_hashed_password))
            data = cursor.fetchall()

            if len(data) is 0:
                conn.commit()
                return json.dumps({'message':'User created successfully !'})
            else:
                return json.dumps({'error':str(data[0])})
        else:
            return json.dumps({'html':'<span>Enter the required fields</span>'})

    except Exception as e:
        return json.dumps({'error':str(e)})
    finally:
        cursor.close() 
        conn.close()


@app.route('/showSignIn')
def showSignin():
    return render_template('signin.html')


@app.route('/validateLogin',methods=['POST'])
def validateLogin():
    try:
        _username = request.form['inputEmail']
        _password = request.form['inputPassword']

        # connect to mysql
        con = mysql.connect()
        cursor = con.cursor()
        cursor.callproc('sp_validateLogin',(_username,))
        data = cursor.fetchall()
 
        if len(data) > 0:
            if check_password_hash(str(data[0][3]),_password):
                session['user'] = data[0][0]
                return redirect('/userHome')
            else:
                return render_template('error.html',error = 'Wrong Email address or Password.')
        else:
            return render_template('error.html',error = 'Wrong Email address or Password.')
 
    except Exception as e:
        return render_template('error.html',error = str(e))
    finally:
        cursor.close()
        con.close()

@app.route('/userHome')
def userHome():
    if session.get('user'):
        conn = mysql.connect()
        cursor = conn.cursor()
        user_id = str(session['user'])
        cursor.execute('SELECT user_name FROM tbl_user WHERE user_id = ' + user_id)
        user_name = cursor.fetchone()
        return render_template('userHome.html', user_name=user_name[0])
    else:
        return render_template('error.html',error = 'Unauthorized Access')


@app.route('/userHome/analysis', methods=['POST', 'GET'])
def analysis():
    if session.get('user'):
        conn = mysql.connect()
        cursor = conn.cursor()
        user_id = str(session['user'])
        cursor.execute('SELECT user_name FROM tbl_user WHERE user_id = ' + user_id)
        user_name = cursor.fetchone()
        ticker = request.form['inputTicker']
        print('Analyzing: ' + ticker)
        head = 'Summary Statistics for ' + ticker + ': '
        predicted_price_time_averaged_5 = str(predict_price_time_averaged(ticker, 2, verbose=0))
        analysis = head + str(get_summary_statistics(ticker))#get_analysis_text(ticker)
        return render_template('userHome.html', user_name=user_name[0], 
            report=analysis, 
            predicted_price=predicted_price_time_averaged_5,
            ticker=ticker
        )
    else:
        return render_template('error.html',error = 'Unauthorized Access')


@app.route('/dashboard')
def show_dashboard():
    return render_template('dashboard.html')



@app.route('/logout')
def logout():
    session.pop('user',None)
    return redirect('/')

if __name__ == "__main__":
    app.run()