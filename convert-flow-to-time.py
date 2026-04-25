import math

def calculate_travel_time_from_traffic_flow(predicted_flow, distance_km, is_congested=False, intersection_delay_seconds=30.0):
    """
    Converts predicted traffic flow (vehicle count) into estimated travel time in minutes.
    returns estimated travel time in minutes
    
    predicted_traffic_flow: predicted vehicles per hour
    distance_km: distance between two SCATS sites in kilometres
    """        
    # The absolute peak of the parabola is 1500. If model predicts > 1500, cap it
    flow = min(predicted_flow, 1500.0)
    
    # If road not congested, and if flow is at or below 351 the speed is capped at the 60 km/h speed limit
    if flow <= 351.0 and not is_congested:
        speed_kmh = 60.0
        return round(((distance_km / speed_kmh) * 60.0) + (intersection_delay_seconds / 60.0), 2)
    
    # TO SOLVE FOR SPEED, REARRANGE THE FORMULA BY MOVING 'flow' OVER TO THE OTHER SIDE SO THAT EQUATION = 0
    # original formula: flow = -1.4648375*(speed)^2 + 93.75*(speed)
    # rearranged formula: -1.4648375*(speed^2) + 93.75*(speed) - flow = 0
    # a*v^2 + b*v + c = 0
    A = -1.4648375
    B = 93.75
    C = -flow
    
    # QUADRATIC FORMULA TO SOLVE FOR SPEED
    # alculate the discriminant of the quadratic formula first
    discriminant = (B ** 2) - (4 * A * C)
    
    # DETERMINE GREEN VS RED LINE SPEED
    if is_congested:
        # RED LINE (Over Capacity / Congested): Slower speed
        # We use the plus (+) sign. Divided by negative A, it gives a smaller positive number.
        speed_kmh = (-B + math.sqrt(discriminant)) / (2 * A)
    else:
        # GREEN LINE (Under Capacity): Faster speed
        # We use the minus (-) sign. Divided by negative A, it gives a larger positive number.
        speed_kmh = (-B - math.sqrt(discriminant)) / (2 * A)
    
    # Prevent division by zero
    speed_kmh = max(0.1, speed_kmh)
    
    # Time = Distance / Speed (Multiply by 60 for minutes)
    travel_time_mins = (distance_km / speed_kmh) * 60.0
    travel_time_mins += intersection_delay_seconds / 60.0
    
    return round(travel_time_mins, 2)

# Test
if __name__ == "__main__":
    print("--- Assignment 2B PDF Math Test ---")
    test_distance = 5.0 # 5km road segment
    
    # Testing the Green Line (Default)
    print("GREEN LINE SCENARIOS (Under Capacity):")
    print(f"  Low Flow (100) -> Expected capped speed (60km/h): {calculate_travel_time_from_traffic_flow(100, test_distance)} mins")
    print(f"  Medium Flow (800) -> Expected normal speed: {calculate_travel_time_from_traffic_flow(800, test_distance)} mins")
    
    # Testing the Red Line
    print("\nRED LINE SCENARIOS (Over Capacity / Traffic Jam):")
    print(f"  Low Flow (100) -> Expected crawling speed: {calculate_travel_time_from_traffic_flow(100, test_distance, is_congested=True)} mins")
    print(f"  Medium Flow (800) -> Expected slow speed: {calculate_travel_time_from_traffic_flow(800, test_distance, is_congested=True)} mins")
    
    # Max Capacity (Should be identical for both)
    print("\nMAX CAPACITY (1500 veh/hr):")
    print(f"  Green Line -> {calculate_travel_time_from_traffic_flow(1500, test_distance)} mins")
    print(f"  Red Line   -> {calculate_travel_time_from_traffic_flow(1500, test_distance, is_congested=True)} mins")