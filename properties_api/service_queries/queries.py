queries= {
    
    "get_all_polygons": '''
        SELECT name, ciudad, ST_AsGeoJSON(poligono), id FROM 
        db.departments_polygons;
    ''',
    "get_polygon_by_city_name": '''
        SELECT name, ciudad, ST_AsGeoJSON(poligono), id FROM 
        db.departments_polygons
        WHERE ciudad = %s;
    '''
    
    
    
}