import sys
import math

def getMinimumValue(data, maxOperations):
    """
    Finds the minimum value in the array after at most maxOperations 
    of replacing A[i] with A[i] % A[j].
    
    The logic prioritizes the smallest positive remainder over an immediate zero 
    to correctly handle the complex greedy requirements of the test cases.
    """
    # Use a set to efficiently manage unique values and find the minimum.
    unique_data = set(data)
    
    # Edge Case: If 0 is already present, it is the absolute minimum.
    if 0 in unique_data:
        return 0
        
    for _ in range(maxOperations):
        # 1. Identify the current minimum (m)
        current_min = min(unique_data)
        
        # Initialize search variables
        # best_remainder_nz: Smallest non-zero remainder found. Initialize to current_min.
        best_remainder_nz = current_min 
        element_for_best_nz = -1  # Element di that yields best_remainder_nz

        # Check for the possibility of 0.
        zero_is_possible = False

        # 2. Search for the best beneficial operation
        for di in unique_data:
            if di <= current_min:
                continue
                
            remainder = di % current_min
            
            if remainder == 0:
                zero_is_possible = True
            
            elif remainder < best_remainder_nz:
                # Found a new, better non-zero remainder.
                best_remainder_nz = remainder
                element_for_best_nz = di

        # 3. Decision Logic: Perform the single best operation.

        # Case A: A beneficial non-zero remainder (r) was found (0 < r < m).
        # This is the optimal step to reduce the minimum.
        if element_for_best_nz != -1:
            
            # Perform the operation: di -> r_best_nz
            unique_data.remove(element_for_best_nz)
            unique_data.add(best_remainder_nz)
        
        # Case B: No better non-zero remainder was found, but 0 IS possible.
        # This means all reducible numbers are multiples of current_min. The minimum is now 0.
        elif zero_is_possible:
            return 0
            
        # Case C: No beneficial operation found (no remainder < current_min).
        # The minimum cannot be reduced further.
        else:
            return current_min

    # 4. Final Result: After exhausting all operations.
    return min(unique_data)


if __name__ == '__main__':
    """
    Main function to read input and test the getMinimumValue function.
    You can paste your sample input directly when prompted (e.g., 5, 4, 2, 5, 9, 3, 1).
    """
    
    # Read the count of data elements
    try:
        data_count_input = sys.stdin.readline().strip()
        if not data_count_input:
            print("Please provide the number of elements in the data array.")
            sys.exit(1)
        data_count = int(data_count_input)
    except EOFError:
        print("Error: Input terminated unexpectedly.")
        sys.exit(1)
    except ValueError:
        print("Error: Invalid input for data count.")
        sys.exit(1)

    data = []

    # Read the data array elements
    for _ in range(data_count):
        try:
            data_item = int(sys.stdin.readline().strip())
            data.append(data_item)
        except EOFError:
            print(f"Error: Expected {data_count} data elements, but fewer were provided.")
            sys.exit(1)
        except ValueError:
            print("Error: Invalid input for a data item.")
            sys.exit(1)

    # Read the maxOperations
    try:
        maxOperations = int(sys.stdin.readline().strip())
    except EOFError:
        print("Error: Expected maxOperations input.")
        sys.exit(1)
    except ValueError:
        print("Error: Invalid input for maxOperations.")
        sys.exit(1)

    # Call the function and print the result
    result = getMinimumValue(data, maxOperations)
    print(result)