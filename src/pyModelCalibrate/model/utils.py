
def strict_lower_bound(x: float, arr: list) -> int:

    start = 0
    end = len(arr) - 1

    while start <= end:

        mid = (start + end) // 2

        if arr[mid] < x:
            start = mid + 1
        else:
            end = mid - 1

    return end
            
    