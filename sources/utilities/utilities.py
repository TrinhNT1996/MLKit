def complement_elements(list1, list2):
    results = []
    for item in list1:
        if item not in list2:
            results.append(item)
    return results
