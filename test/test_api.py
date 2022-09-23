from fastapi.testclient import TestClient
import sys
import os
from main import app

absolute_path = os.path.dirname(os.path.abspath(__file__))
path_ab = os.path.join(absolute_path, '../train_script')
sys.path.append(path_ab)

print("[info] working dir is {0}".format(path_ab))

client_app = TestClient(app)
print("[info] creating the working app")


def test_get_method():
    print("[info] check get method return true string")
    r = client_app.get("/")
    assert r.json() == "[Info] This Get method for FastAPI inference."
    assert r.status_code == 200

    print("[info] check get method return false string")
    r = client_app.get("/check")
    assert r.status_code != 200


def test_post_greater_50k(data_frame_greater_than_50k):

    print("[info] check post method return output which is greater than 50k")
    request = client_app.post('/inference', json=data_frame_greater_than_50k)
    assert request.status_code == 200
    assert "output" in request.json()
    assert request.json()["output"] == ">50K"


def test_post_smaller_50k(data_frame_smarter_than_50k):

    print("[info] check post method return output which is smaller than 50k")
    request = client_app.post('/inference', json=data_frame_smarter_than_50k)
    assert request.status_code == 200
    assert "output" in request.json()
    assert request.json()["output"] == "<=50K"
