import os  
import psycopg2
import importlib
importlib.reload(psycopg2)
from psycopg2 import sql  
  
# CSV文件所在的目录  
csv_dir = r'D:\something\database\database1_dataset\姓名判断性别'  
schema_name = 'schemas'  

# 连接到数据库  
conn = psycopg2.connect(dbname='databases',user='postgres',password='qazxde123',host='127.0.0.1',port='5432')
cur = conn.cursor()  

# 设置搜索路径  
cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name)))  
  
# 遍历CSV目录  
for filename in os.listdir(csv_dir):  
    if filename.endswith('.csv'):  
        table_name = filename[:6]  
        file_path = os.path.join(csv_dir, filename)  
  
        # 创建表（如果不存在）  
        create_table_sql = sql.SQL("""  
            CREATE TABLE IF NOT EXISTS {table_name} (  
                {table_name}_id SERIAL PRIMARY KEY,  
                DATE DATE,  
                OPEN NUMERIC,  
                END NUMERIC,  
                MAX NUMERIC,  
                MIN NUMERIC,  
                VOL BIGINT,  
                TV NUMERIC,  
                VAR NUMERIC,  
                CHANGE_PCT NUMERIC,  
                CHANGE_AMT NUMERIC,  
                TR NUMERIC  
            )  
        """).format(table_name=sql.Identifier(table_name))  
        cur.execute(create_table_sql)  
  
        # 执行COPY命令  
        copy_sql = sql.SQL("""  
            COPY {table_name} (DATE,OPEN,END,MAX,MIN,VOL,TV,VAR,CAHNGE_PCT,CHANGE_AMT,TR)  
            FROM STDIN WITH (FORMAT csv, HEADER)  
        """).format(table_name=sql.Identifier(table_name))  
          
        with open(file_path, 'r') as f:  
            cur.copy_expert(copy_sql, f)  
  
# 提交事务并关闭连接  
conn.commit()  
cur.close()  
conn.close()