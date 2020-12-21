# Serverless ML Inferencing with AWS Lambda and TensorFlow Lite

by Sandeep Mistry (on behalf of [Assent Compliance Inc.](https://www.assentcompliance.com))

> ... A lot of the tutorials you'll find on "how to get started with TensorFlow machine learning" talk about training models. Many even just stop there once you've trained the model and then never ever touch it again. ...

-- [Alasdair Allan](https://twitter.com/aallan) @ [QCon 2020](https://www.infoq.com/presentations/iot-edge-ml-privacy-security/)

In this tutorial we'll be walking through how to train a machine learning model using the [AutoKeras](https://autokeras.com) library, converting the model to [TensorFlow Lite](https://www.tensorflow.org/lite) format, and deploying it to an [AWS Lambda](https://aws.amazon.com/lambda/) environment for highly cost effective and scalable ML inferencing.

While [AWS SageMaker](https://aws.amazon.com/sagemaker/) provides a convenient mechanism to deploy TensorFlow models as a REST API using [TensorFlow Serving Endpoints](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html), this method does not suit the needs of Assent's Intelligent Document Analysis pipeline, as we are dealing with unpredictable spiky traffic loads. For more information on this project, checkout the recording of the [Intelligent Document Analysis webinar](https://youtu.be/u1ZK2jZx5D0) my colleague Corey Peters co-presented with the AWS team in September 2020.

We decided to investigate if it was possible to perform our image and text classification inferencing in an AWS Lambda environment to both reduce our monthly AWS bill and improve system scalability. AWS Lambda currently has a code size limit of 250 MB per Lambda (for non-Docker containers), and this is significantly smaller than the installed size of the [TensorFlow](https://www.tensorflow.org) Python package with dependencies (900+ MB). It is possible to use [Amazon EFS](https://aws.amazon.com/efs/) for AWS Lambda with the regular TensorFlow Python package, as described in [AWS's Building deep learning inference with AWS Lambda and Amazon EFS blog post](https://aws.amazon.com/blogs/compute/building-deep-learning-inference-with-aws-lambda-and-amazon-efs/) however, this approach can have a relatively large ["cold start"](https://mikhail.io/serverless/coldstarts/define/) time for your AWS Lambda.

The [TensorFlow Lite](https://www.tensorflow.org/lite) project's tag line is:

> "Deploy machine learning models on mobile and IoT devices". 

-- https://www.tensorflow.org/lite

It provides a much smaller runtime focused on ML inferencing on mobile and IoT devices which have more constrained computing environments (CPU, RAM, storage) than cloud computing environments. Luckily TensorFlow Lite provides a [Python based runtime API](https://www.tensorflow.org/lite/guide/python) for Linux based embedded computers. After some experimentation, we found it is possible to re-use TensorFlow Lite Python run-time in a AWS Lambda environment. There were a few key steps:

1. Creating a custom AWS Lambda layer for the TensorFlow Lite Python runtime. This involved compiling the TensorFlow source code to create the TF Lite Python runtime in the AWS Lambda Python [Docker](https://www.docker.com) build container. With dependencies, the installed package size for this is 53 MB, which is approximately 17x smaller than regular TensorFlow!

2. Converting our [Keras](https://keras.io) or TensorFlow based models to TensorFlow Lite format.


I'll be walking through an example of this below. To keep the ML model training side simple we'll be using the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) dataset along with the [AutoKeras Image Classification tutorial](https://autokeras.com/tutorial/image_classification/) to create the model we want to deploy to an AWS Lambda environment. The goal will be to create a REST API that accepts an input image and returns a prediction on what digit is contained in this image along with a confidence score for this prediction.


You'll need a computer with the following setup:
 - Python 3.8
 - Docker Desktop
 - an [AWS account](https://aws.amazon.com) (if you wish to deploy the REST API)

## Training our ML model

This section is based on the awesome AutoKeras [Image Classification Tutorial](https://autokeras.com/tutorial/image_classification/).

We'll start off by installing AutoKeras using `pip`:

```bash
pip install autokeras
```

Now let's train and evaluate our model:

```python
# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# Initialize the image classifier.
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1)
# Feed the image classifier with training data.
clf.fit(x_train, y_train, epochs=10)

# Predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)

# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
```

Now that the model has been trained, let's [export](https://autokeras.com/tutorial/export/) and save it as a Keras model:

```python
# export model
model = clf.export_model()

# save Keras model to disk
model.save('keras_image_classifier')
```

## Converting the model to TF Lite format

Now that we have a Keras model, we can follow the [TensorFlow Lite converter - Convert a Keras model](https://www.tensorflow.org/lite/convert#convert_a_keras_model_) guide to convert it to `.tflite` format.

We'll save the `.tflite` file to the `src` folder where our AWS Lambda source code is stored.

```python
# Convert to TFLite
# https://www.tensorflow.org/lite/convert#convert_a_keras_model_
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# Save the model.
with open('src/model.tflite', 'wb') as output:
    output.write(tflite_model)
```

## Testing out the TF Lite model

Let's create a Python `class` to wrap the TensorFlow runtime so the API is more like the AutoKeras API.

We'll create a new file in `src/image_classifier.py`:

```python
# https://www.tensorflow.org/lite/guide/python
try:
    # try TF Lite Runtime first
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    
import numpy as np

class ImageClassifier:
    def __init__(self, model_filename='model.tflite'):
        # load the .tflite model with the TF Lite runtime interpreter
        self.interpreter = Interpreter(model_filename)
        self.interpreter.allocate_tensors()

        # get a handle to the input tensor details
        input_details = self.interpreter.get_input_details()[0]
        
        # get the input tensor index
        self.input_index = input_details['index']

        # get the shape of the input tensor, so we can rescale the
        # input image to the appropriate size when making predictions
        input_shape = input_details['shape']
        self._input_height = input_shape[1]
        self._input_width = input_shape[2]

        # get a handle to the input tensor details
        output_details = self.interpreter.get_output_details()[0]
        
        # get the output tensor index
        self.output_index = output_details['index']

    def predict(self, image):   
        # convert the image to grayscale and resize
        grayscale_image = image.convert('L').resize((self._input_width, self._input_height))

        # convert the image to a numpy array
        input = np.asarray(grayscale_image.getdata(), dtype=np.uint8).reshape((1, self._input_width, self._input_height))

        # assign the numpy array value to the input tensor
        self.interpreter.set_tensor(self.input_index, input)

        # invoke the operation
        self.interpreter.invoke()

        # get output tensor value
        output = self.interpreter.get_tensor(self.output_index)

        # return the prediction, there was only one input
        return output[0]

    def classify(self, image):
        # get the prediction, output with be array of 10
        prediction = self.predict(image)
        
        # find the index with the largest value
        classification = int(np.argmax(prediction))
        
        # get the score for the largest index
        score = float(prediction[classification])

        return classification, score
```

Now let's try out the class with a test image. First we'll need to install the Python [Pillow](https://python-pillow.org) package.

```bash
pip install Pillow
```


```python
# import the Pillow module to read test images from disk
from PIL import Image

# import image classifier class from previous step
from src.image_classifier import ImageClassifier

# load test image from disk
zero_image = Image.open('test_images/0.png')

# create image classifier instance with path to TF Lite model file
image_classifier = ImageClassifier('src/model.tflite')

# classify the image
classification, score = image_classifier.classify(zero_image)

print(f'Image classified as {classification} with score {score}')
```

## Building the TF Lite runtime .whl package

The TensorFlow Lite team provides [pre-built](https://www.tensorflow.org/lite/guide/python) Python `.whl` packages for various architectures including Linux (x86-64), but unfortunately the AWS Lambda 3.8 Python environment is not compatible with the pre-built Python Linux (x86-64) package.

If you try to run:
```
pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.4.0-cp38-cp38-linux_x86_64.whl
```

The following error will result:
```
...
aws_lambda_builders.workflows.python_pip.packager.PackageDownloadError: ERROR: tflite_runtime-2.4.0-cp38-cp38-linux_x86_64.whl is not a supported wheel on this platform
...
```

However, we can use some information from the [Build TensorFlow Lite for Raspberry Pi](https://www.tensorflow.org/lite/guide/build_rpi) to build our own `.whl` file for the AWS Lambda's Python 3.8 environment.

We'll use the following `Dockerfile` for this:


```docker
# Use https://hub.docker.com/r/lambci/lambda as the base container
FROM lambci/lambda:build-python3.8 AS stage1

# set the working directory to /build
WORKDIR /build

# download Bazel (used to compile TensorFlow)
RUN curl -L https://github.com/bazelbuild/bazel/releases/download/3.7.1/bazel-3.7.1-linux-x86_64 -o /usr/bin/bazel && chmod +x /usr/bin/bazel

# make python3 the default python
RUN ln -sf /usr/bin/python3 /usr/bin/python
    
# Use git to clone the TensorFlow source, checkout v2.4.0 branch
RUN git clone https://github.com/tensorflow/tensorflow.git --branch v2.4.0 --depth 1

# install TensorFlow Lite Python dependencies
RUN pip3 install pybind11 numpy

# start the TensorFlow Lite build with Bazel
RUN BAZEL_FLAGS='--define tflite_with_xnnpack=true' ./tensorflow/tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh

# copy the built TensorFlow Lite Python .whl file to the Docker host
FROM scratch AS export-stage
COPY --from=stage1 /build/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.4.0-py3-none-any.whl .

```

Now we'll kick off the `docker build`. The build `.whl` file will be stored in `layers/tflite_runtime/`, for use in the next step.

```bash
DOCKER_BUILDKIT=1 docker build --output layers/tflite_runtime/ .
```

## Create the AWS Serverless Application Model (SAM) infrastructure

> The AWS Serverless Application Model (SAM) is an open-source framework for building serverless applications.

-- https://aws.amazon.com/serverless/sam/

We'll use AWS's `sam` CLI tool for the structure of our Serverless Application, and install it via `pip`:

```bash
pip install aws-sam-cli
```

Our SAM app will have the following folder structure:

- `events/` - contains test events as JSON files
- `layers/` - contains pre-built Python packages the AWS Lambda depends on
- `src/` - contains source code for our AWS Lambda
- `template.yaml` - defines the structure of our SAM application

### `template.yaml`

Now let's look at the `template.yaml`, it defines:
- a `Global` section that defines:
  - The default AWS Lambda Function timeout in seconds
  - The default list of media types the Serverless API should treat as binary content types
  
- a `Resources` section that defines:
  - The AWS Lambda Function that will handle the ML Inferencing
  - A Serverless API gateway
  - AWS Lambda layer that contains the Python TensorFlow Lite runtime
  - AWS Lambda layer that contains the Python [Pillow](https://python-pillow.org) package used for processing Images
  
- an `Outputs` section that defines fields to be exported one this SAM app is deployed to AWS

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Sample SAM TF Lite Function

# More info about Globals: https://github.com/awslabs/serverless-application-model/blob/master/docs/globals.rst
Globals:
  Function:
    Timeout: 30
  Api:
    BinaryMediaTypes:
    - image~1png
    - image~1jpeg

Resources:
  TFLiteFunction:
    Type: AWS::Serverless::Function # More info about Function Resource: https://github.com/awslabs/serverless-application-model/blob/master/versions/2016-10-31.md#awsserverlessfunction
    Properties:
      CodeUri: src/
      Layers:
        - !Ref TFLiteRuntimeLayer
        - !Ref PillowLayer
      Handler: app.lambda_handler
      Runtime: python3.8
      Events:
        Base:
          Type: Api
          Properties:
            Method: any
            Path: /
            RestApiId: !Ref ServerlessRestApi
        Others:
          Type: Api
          Properties:
            Method: any
            Path: /{proxy+}
            RestApiId: !Ref ServerlessRestApi
            
  # https://aws.amazon.com/blogs/developer/handling-arbitrary-http-requests-in-amazon-api-gateway/
  ServerlessRestApi:
    Type: "AWS::Serverless::Api"
    Properties:
      StageName: Prod
      DefinitionBody:
        openapi: "3.0"
        info:
          title: !Ref "AWS::StackName"
          version: "1.0"
        paths:
          /:
            x-amazon-apigateway-any-method:
              responses:
                {}
            x-amazon-apigateway-integration:
              httpMethod: POST
              type: aws_proxy
              uri: !Sub "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${TFLiteFunction.Arn}/invocations"
          /{proxy+}:
            x-amazon-apigateway-any-method:
              responses:
                {}
            x-amazon-apigateway-integration:
              httpMethod: POST
              type: aws_proxy
              uri: !Sub "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${TFLiteFunction.Arn}/invocations"


  # https://aws.amazon.com/blogs/compute/working-with-aws-lambda-and-lambda-layers-in-aws-sam/
  TFLiteRuntimeLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: TFLiteRuntime
      Description: TF Lite Runtime Layer
      ContentUri: layers/tflite_runtime/
      CompatibleRuntimes:
        - python3.8
      RetentionPolicy: Retain
    Metadata:
      BuildMethod: python3.8
            
  PillowLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: Pillow
      Description: Pillow Layer
      ContentUri: layers/pillow/
      CompatibleRuntimes:
        - python3.8
      RetentionPolicy: Retain
    Metadata:
      BuildMethod: python3.8

Outputs:
  # ServerlessRestApi is an implicit API created out of Events key under Serverless::Function
  # Find out more about other implicit resources you can reference within SAM
  # https://github.com/awslabs/serverless-application-model/blob/master/docs/internals/generated_resources.rst#api
  TFLiteApi:
    Description: "API Gateway endpoint URL for Prod stage for TF Lite function"
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/"
  TFLiteFunction:
    Description: "TF Lite Lambda Function ARN"
    Value: !GetAtt TFLiteFunction.Arn
  TFLiteFunctionIamRole:
    Description: "Implicit IAM Role created for TF Lite function"
    Value: !GetAtt TFLiteFunctionRole.Arn
```

### `src/` folder

Now let's go over the files in the `src/` folder.

The `app.py` file contains the Python code for our Lambda.

It will receive an event for the HTTP POST request, which contains an image, and processes it:

1. Verifies the body is base64 encoded (via the API Gateway proxy)
2. Decodes the base64 data to bytes
3. Uses the image bytes with `Pillow` to create an Image object
4. Uses the `image_classifier` instance to classify the image
5. Returns a response with a JSON body for the API Gateway to return to the REST API client 

```python
import base64
import json

from io import BytesIO
from PIL import Image

from image_classifier import ImageClassifier

# create an Image Classifier instance
image_classifier = ImageClassifier()
        
def lambda_handler(event, context):
    if event['isBase64Encoded'] is not True:
        return {
            'statusCode': 400 # Bad Request!
        }
    
    
    # The event body will have a base64 string containing the image bytes,
    # Deocde the base64 string into bytes
    image_bytes = base64.b64decode(event['body'])

    # Create a Pillow Image from the image bytes
    image = Image.open(BytesIO(image_bytes))
    
    # Use the image classifier to get a prediction for the image
    value, score = image_classifier.classify(image)
    
    # return the response as JSON
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
        },
        "body": json.dumps({
            'value': value,
            'score': score 
        })
    }
```

The `sam` CLI requires a `requirements.txt` file in the Python Lambda's in source code folder. We can create an empty file.

```bash
touch src/requirements.txt
```

In earlier steps we've already placed the necessary `image_classifier.py` and `model.tflite` files in the `src/` folder.

### `layers/` folder

This folder will contain files to generate the AWS Lambda Layers that will contain the third party Python modules our AWS Lambda depends on.

For the `layers/pillow` folder with just need a `requirements.txt` file a single line:

```
Pillow==8.0.1
```

Similarly for the `layers/tflite_runtime` folder we need a `requirements.txt` file a single line: 

```
tflite_runtime-2.4.0-py3-none-any.whl
```

Unlike the `layers/pillow/requirements.txt` file which specified the Python module name and version, this file will refer to the `.whl` file we built in an earlier step.

## Building the SAM application

We can now use the AWS SAM CLI to build the application, and we'll use the `--use-container` option so that it builds the AWS Lambda Layers defined in the `template.yaml` inside a `Docker` container:

```bash
sam build --use-container
```

Now that our AWS SAM app has been built locally, we can test it out with a test event.

The `events/0.json` file contains a Base64 encoded image of a `0`.

The expected outcome of this will be a prediction for "0":

```bash
sam local invoke TFLiteFunction --event events/0.json
```

You can try out the other test JSON event files in the `events` folder to ensure the ML model is performing as expected. The `test_images` folder contains the original test images in PNG format.

## Deploying to AWS

We can now deploy the system to AWS, and the `sam` CLI tool can be used for this.

Make sure to replace the `<stack name>` and `<region>` in the command below:

```bash
sam deploy --stack-name my-tflite-app --resolve-s3 --capabilities CAPABILITY_IAM --region us-east-1
```

Now that the app is deployed to AWS we can test it out with the `curl` command.

Make sure to replace the URL in the command below with the one outputted from the previous step.

```bash
curl -X POST -H 'Content-Type: image/png' --data-binary @'test_images/0.png' https://<api id>.execute-api.<region>.amazonaws.com/Prod/
```

## Conclusion

In this tutorial we covered:

1. Training an image classification model using the AutoKeras model to classify handwritten digits.
2. Converting the model to TensorFlow Lite format.
3. Building the TensorFlow Lite Python runtime for usage in AWS Lambda.
4. Creating an AWS SAM app that uses the TensorFlow Lite model with the TensorFlow Lite Python runtime to classify an image received over an HTTP API.

While a simple model was trained and deployed, you can follow the same process for other models that are trained in TensorFlow and compatible with the TensorFlow Lite runtime.
