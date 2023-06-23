from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, Polygon, Feature
import json


def parse_polygons(data_polygon, type_of_parse=None) -> list:
    if type_of_parse == "all_polygons_array_parsing":
        dict_final_polygons = {}
        for polygon in data_polygon:
            coordinates = json.loads(polygon[2]).get("coordinates")
            id_polygon = polygon[3]
            array_coordinates = coordinates[0][0]
            polygon = [  {"lat":point[1] , "lng" :point[0]}   for point in array_coordinates]
            dict_final_polygons[id_polygon] = polygon
        return dict_final_polygons
    coordinates = json.loads(data_polygon[2]).get("coordinates")
    if not coordinates:
        return {}
    array_coordinates = coordinates[0][0]
    if type_of_parse == "find_polygon_by_point_parsing":
        polygon = [ (point[0] , point[1])   for point in array_coordinates]
        return polygon
    polygon = [  {"lat":point[1] , "lng" :point[0]}   for point in array_coordinates]
    return polygon

def find_polygon_by_point(coordinates_point, array_data_polygon) -> bool:
    lng = float(coordinates_point["lng"])
    lat = float(coordinates_point["lat"])
    point = Feature(geometry=Point((lng, lat)))
    dict_polygons_validations = {}
    for data_polygon in array_data_polygon:
        name_polygon = data_polygon[0].replace("Ã‘","N")
        dict_polygons_validations[name_polygon] ={}
        dict_polygons_validations[name_polygon]["id"] = data_polygon[3]
        dict_polygons_validations[name_polygon]["match"] = False
        
        data_polygon = parse_polygons(data_polygon=data_polygon, type_of_parse="find_polygon_by_point_parsing")
        polygon = Polygon(
            [
                data_polygon
            ]
        )
        result = boolean_point_in_polygon(point, polygon)
        dict_polygons_validations[name_polygon]["match"] = result
    final_array_result = []
    for k , v in dict_polygons_validations.items():
        if v["match"] is True:
            final_array_result.append({k:v})
    return final_array_result