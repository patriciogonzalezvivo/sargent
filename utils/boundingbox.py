def boundingBox(points):
    min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
    max_x, max_y,max_z = float("-inf"), float("-inf"), float("-inf")
    for point in points:
        if point[0] < min_x:
            min_x = point[0]
        if point[1] < min_y:
            min_y = point[1]

        if point[0] > max_x:
            max_x = point[0]
        if point[1] > max_y:
            max_y = point[1]

        if len(point) == 3:
            if point[2] < min_z:
                min_z = point[2]
            if point[2] > max_z:
                max_z = point[2]
    
    if len(point) == 3:
        return min_x, min_y, min_z, max_x, max_y, max_z
    else:
        return min_x, min_y, max_x, max_y