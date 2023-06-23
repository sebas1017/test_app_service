from dotenv import load_dotenv, dotenv_values
from service_queries.queries import queries
load_dotenv()
config = dotenv_values(".env")
import mysql.connector

def open_connection():
   
    return cursor

def generic_querie(sql_querie, config=config):
    try:
        engine = mysql.connector.connect(
            host=config.get('mysql_host'),
            user=config.get('mysql_username'),
            password=config.get('mysql_password'),
            database=config.get('mysql_dbname')
            )
        cursor = engine.cursor()
        querie = queries.get(sql_querie)
        if not querie:
            return "querie doesn't exist please verify"
        cursor.execute(querie)
        data = cursor.fetchall()
        if not data:
            return {}
        return data
    except Exception as e:
        print(f"Exception in generic_querie db -> {e}")


def querie_filter_polygon_by_city(sql_querie, city_name):
    try:
        engine = mysql.connector.connect(
                host=config.get('mysql_host'),
                user=config.get('mysql_username'),
                password=config.get('mysql_password'),
                database=config.get('mysql_dbname')
                )
        cursor = engine.cursor()
        querie = queries.get(sql_querie)
        if not querie:
            return "querie doesn't exist please verify"
        cursor.execute(querie, (city_name,))
        data = cursor.fetchall()
        if not data:
            return {}
        return data
    except Exception as e:
        print(cursor._executed.decode())
        print(f"Exception in querie_filter_polygon_by_city db -> {e}")