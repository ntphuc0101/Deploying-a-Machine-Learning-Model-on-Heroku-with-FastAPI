import requests


def query_api(url, data):
    print("[info] here is the url {0}", url)
    print("[info] here is the data {0}", data)
    output_request = requests.post(url, json=data)
    if output_request.status_code == 200:
        return output_request.status_code, output_request.json()
    return output_request.status_code, None


def data_frame_greater_than_50k():
    df_data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native_country": "United-States"
    }
    return df_data

def data_frame_smarter_than_50k():
    df_data = {
        "age": 28,
        "workclass": "Private",
        "fnlgt": 338409,
        "education": "Bachelors",
        "education-num": 20,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native_country": "India"
    }
    return df_data

if __name__ == '__main__':
    url_request = "https://herokuwithfastapi.herokuapp.com/inference"
    print("url app {0}", url_request)
    print("[info] testing dataframe smaller than 50k")
    df_frame = data_frame_smarter_than_50k()


    status_code, predicted = query_api(url_request, df_frame)
    print("[info] STATUS CODE {0}", status_code)
    print("[info] STATUS CODE {0}", predicted)


    print("[info] testing dataframe greater than 50k")
    df_frame = data_frame_greater_than_50k()

    status_code, predicted = query_api(url_request, df_frame)
    print("[info] STATUS CODE {0}", status_code)
    print("[info] STATUS CODE {0}", predicted)