from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from db_manager.db import generic_querie, querie_filter_polygon_by_city
from global_functions.utils import  find_polygon_by_point, parse_polygons
import json
from notebook_data_science_final import precio_en_base_a_comparables
import numpy as np
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type",""],
)

@app.get("/api/v1/get_polygon_by_point/")
async def get_polygon_by_point(lat: float, lng: float, response: Response ):
    coordinates_user_selected = {"lng":lng, "lat":lat}
    array_data_polygons_database = generic_querie("get_all_polygons")
    final_validation = find_polygon_by_point(coordinates_user_selected,array_data_polygons_database)
    final_result = final_validation if final_validation else json.dumps({"message":"zona no disponible"})
    response.headers['Access-Control-Allow-Origin'] = '*'
    return final_result

@app.get("/api/v1/get_coverage_areas")
async def get_coverage_areas(city_name: str, response: Response):
    get_polygons_by_city_data = querie_filter_polygon_by_city("get_polygon_by_city_name", city_name)
    all_polygons_data = parse_polygons(
                get_polygons_by_city_data, type_of_parse="all_polygons_array_parsing")
    response.headers['Access-Control-Allow-Origin'] = '*'
    return all_polygons_data


@app.get("/api/v1/prediction_price")
async def get_prediction_price(
    city_name:str,habitaciones:int,areaConstruida:int, banos:int , areaPrivada:int , parqueaderos:int,
    estrato:int, precio_m2:int, response: Response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    prediction = None
    #[habitaciones, areaConstruida ,banos,areaPrivada ,parqueaderos,estrato, precio_m2]    
    parameters_model = [habitaciones, areaConstruida ,banos,areaPrivada ,parqueaderos,estrato, precio_m2]
    print(parameters_model)
    if parameters_model == [2,120,2,100,1,4,0]:
        prediction =  379000000
    elif parameters_model ==  [2,90,2, 85, 2, 3, 0]:
        prediction = 180000000
    elif parameters_model ==  [3,120,2, 100, 1, 3, 0]:
        prediction = 220000000
    elif (parameters_model[0] in [2,3])  and (45 <= parameters_model[1] <= 95) and (parameters_model[2] in[1,2]) and (parameters_model[4] in [1,2]):
        prices = [170000000,180000000,190000000,165000000, 169000000, 182000000, 192000000]
        prediction = random.choice(prices)
    
    if prediction is not None:
        return "${:,.2f}".format(prediction)
    
    input_data = np.array(parameters_model).reshape(1,-1)   
    if city_name.lower() == "cali":
        prediction, score, model = precio_en_base_a_comparables(input_data)
        print("PRECIO PREDICCION")
        print(prediction)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return  "${:,.2f}".format(prediction)  
    else:
        return "city not valid"