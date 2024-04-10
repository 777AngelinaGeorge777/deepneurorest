#BL.EN.U4CSE21017: Angelina George
from flask import Flask, jsonify, request
import DLModelMLP as DML
import re

app = Flask(__name__)

#/webhook/email
@app.route('/api', methods=['POST'])
def extract_url():
    # Check if the request contains JSON data
    if request.is_json:
        # Parse JSON data from the request body
        data = request.get_json()

        # Check if the 'url' key is present in the JSON data
        if (('url' in data) and ('uid' in data)):
            # Extract the URL from the JSON data
            url = data['url']
            
            uid=data['uid']
            uid=re.sub(r'[^\w_. -]', '_', uid)[0:10]

            # Perform any processing on the extracted URL here
            # For demonstration, we'll simply return the extracted URL
            
            DML.testmodel(url, uid)
            images=DML.plotgraphs(uid)


            return jsonify({'images': images}), 200
        else:
            # If 'url' key is missing, return a response with a 400 status code (Bad Request)
            return jsonify({'error': 'Missing URL in request body'}), 400
    else:
        # If the request does not contain JSON data, return a response with a 400 status code
        return jsonify({'error': 'Invalid request format. Must be JSON'}), 400

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
