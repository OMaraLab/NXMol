def calculate_band_distance(value, lower_bound, upper_bound):
    """
    Calculates the distance of a single value from a defined band.
    If the value is within the band, the distance is 0.
    Otherwise, it's the shortest distance to either bound.

    Args:
        value (float): The individual data point.
        lower_bound (float): The lower limit of the band.
        upper_bound (float): The upper limit of the band.

    Returns:
        float: The distance of the value from the band.
    """
    if lower_bound <= value <= upper_bound:
        return 0.0
    elif value < lower_bound:
        return lower_bound - value
    else: # value > upper_bound
        return value - upper_bound
