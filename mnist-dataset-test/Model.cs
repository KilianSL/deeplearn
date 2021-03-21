using NEA.NeuralNetwork;
using NEA.NeuralNetwork.Layers;
using NEA.Tensor;
using System;

namespace mnist_dataset_test
{
    internal class Model : IModule
    {
        public bool logging; // whether or not to log each layer

        // Layers
        private Layer input = new Linear(784, 2500);

        private Layer h1;
        private Layer h2;
        private Layer h3;
        private Layer h4;
        private Layer output;
        private Layer dropout1000;

        public Model()
        {
            logging = false;
            h1 = new Linear(2500, 2500);
            h2 = new Linear(2500, 1500);
            h3 = new Linear(1500, 1000);
            h4 = new Linear(1000, 500);
            output = new Linear(500, 10);
            dropout1000 = new Dropout(1000, 0.3f);
        }

        public Tensor Forward(Tensor x)
        {
            if (logging) { Console.WriteLine("Starting training. Input size {0}x{1}x{2}", x.Shape[0], x.Shape[1], x.Shape[2]); }
            x = input.Forward(x);
            x = ActivationFunctions.ReLU(x);
            if (logging) { Console.WriteLine("Layer 1. Sample size {0}x{1}x{2}", x.Shape[0], x.Shape[1], x.Shape[2]); }
            x = h1.Forward(x);
            x = ActivationFunctions.ReLU(x);
            if (logging) { Console.WriteLine("Layer 2. Sample size {0}x{1}x{2}", x.Shape[0], x.Shape[1], x.Shape[2]); }
            x = h2.Forward(x);
            x = ActivationFunctions.Sigmoid(x);
            if (logging) { Console.WriteLine("Layer 3. Sample size {0}x{1}x{2}", x.Shape[0], x.Shape[1], x.Shape[2]); }
            x = h3.Forward(x);
            x = dropout1000.Forward(x);
            if (logging) { Console.WriteLine("Layer 4. Sample size {0}x{1}x{2}", x.Shape[0], x.Shape[1], x.Shape[2]); }
            x = ActivationFunctions.Tanh(x);
            x = h4.Forward(x);
            if (logging) { Console.WriteLine("Layer 5. Sample size {0}x{1}x{2}", x.Shape[0], x.Shape[1], x.Shape[2]); }
            x = ActivationFunctions.ReLU(x);
            x = output.Forward(x);
            x = ActivationFunctions.Softmax(x);
            if (logging) { Console.WriteLine("Layer 6. Output size {0}x{1}x{2}", x.Shape[0], x.Shape[1], x.Shape[2]); }
            return x;
        }

        public void setLog(bool state)
        {
            logging = state;
        }
    }
}