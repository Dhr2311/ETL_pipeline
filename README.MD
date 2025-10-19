# ETL pipeline

A ETL piepeline to merge different datasets and make a master table along with an interactive dashboard to browse through it.

Commands to run the pipeline without docker
```
conda create -n ETL python=3.10 
conda activate ETL
pip install -r requirements.txt 
python DataClean.py Patient_data
python etl_from_excel.py  
python app.py  

```

Commands to run the pipeline with docker
```
docker compose up --build  

```

Open: http://127.0.0.1:8050
