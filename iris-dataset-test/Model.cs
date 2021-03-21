using NEA.NeuralNetwork;
using NEA.NeuralNetwork.Layers;
using NEA.Tensor;

namespace iris_dataset_test
{
    internal class Model : IModule
    {
        private Layer inputToHidden;
        private Layer hiddenToOutput;
        private Layer dropout;

        public Model()
        {
            inputToHidden = new Linear(4, 3);
            hiddenToOutput = new Linear(3, 3);
            dropout = new Dropout(3, 0.3f);
        }

        public Tensor Forward(Tensor x)
        {
            x = inputToHidden.Forward(x);
            x = ActivationFunctions.ReLU(x);
            x = dropout.Forward(x);
            x = hiddenToOutput.Forward(x);
            x = ActivationFunctions.Softmax(x);
            return x;
        }
    }
}