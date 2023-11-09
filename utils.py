def convert_to_numbers(str_list):
    converted_list = []
    for item in str_list:
        if item != None:
            try:
                converted_item = int(item)
            except ValueError:
                try:
                    converted_item = float(item)
                except ValueError:
                    converted_item = item

            converted_list.append(converted_item)
        else:
            converted_list.append(None)
    return converted_list