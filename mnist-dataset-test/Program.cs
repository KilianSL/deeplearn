﻿using NEA.Utils.Data;
using System;
using System.Diagnostics;
using NEA.NeuralNetwork;
using System.Linq;

namespace mnist_dataset_test
{
    internal class Program
    {
        // Constants
        private const string DATASET_PATH = @"C:\Users\kilian\source\Git Repos\WCGS-2021-6C-seifk\mnist-dataset-test\mnist.csv";

        private const int BATCH_SIZE = 32;
        private const int N_CLASSES = 10;

        private static string arrayToString(float?[] arr)
        {
            string res = "[";
            foreach (var item in arr)
            {
                res = string.Concat(res, item.ToString() + ',');
            }
            res = string.Concat(res, ']');
            return res;
        }

        private static string arrayToString(float[] arr)
        {
            string res = "[";
            foreach (var item in arr)
            {
                res = string.Concat(res, item.ToString() + ',');
            }
            res = string.Concat(res, ']');
            return res;
        }

        private static void Main(string[] args)
        {
            var timer = new Stopwatch();
            // Calls the dataset test function, plus returns dataset for further use.
            var mnistDataset = TestDataset(timer);
            // Tests the dataloader
            var dataloader = TestDataLoader(timer, mnistDataset);
            // Neural network time
            TestNeuralNetwork(timer, dataloader);
            Console.WriteLine("****** Testing Complete ******");
        }

        private static DataSet TestDataset(Stopwatch timer)
        {
            // DATASET TESTS
            Console.WriteLine("Loading MNIST Data");
            timer.Start();
            var mnistDataset = new DataSet(DATASET_PATH, true);
            timer.Stop();
            Console.WriteLine("Loaded dataset in {0}ms\nSamples: {1}\nFeatures:{2}\n", timer.ElapsedMilliseconds, mnistDataset.Count, mnistDataset.Features);
            timer.Restart();
            var samplesRemoved = mnistDataset.Clean();
            timer.Stop();
            Console.WriteLine("Removed {0} samples (expected 10)\nCleaned dataset in {1}ms\n", samplesRemoved, timer.ElapsedMilliseconds);
            timer.Restart();
            mnistDataset.Shuffle();
            timer.Stop();
            Console.WriteLine("Shuffled dataset in {0}ms\n", timer.ElapsedMilliseconds);
            var sample = mnistDataset.RandomSample(1);
            Console.WriteLine("Got random sample: {0}\n", arrayToString(sample[0]));
            Console.WriteLine("Testing RemoveElementAt() with first element\n\nFirst Element: {0}\n\nRemoving item...\n", arrayToString(mnistDataset[0]));
            timer.Restart();
            mnistDataset.RemoveElementAt(0);
            timer.Stop();
            Console.WriteLine("New first item: {0}\n\nRemoved in {1}ms\n", arrayToString(mnistDataset[0]), timer.ElapsedMilliseconds);
            return mnistDataset;
        }

        private static DataLoader TestDataLoader(Stopwatch timer, DataSet mnistDataset)
        {
            Console.WriteLine("\n****** Commencing Dataloader Tests ******\n");
            Console.WriteLine("Creating Dataloader");
            timer.Restart();
            var dataloader = new DataLoader(mnistDataset, BATCH_SIZE, new int[] { 0 }, true, N_CLASSES, trainTestSplit: 0.1f);
            timer.Stop();
            Console.WriteLine("Created dataloader in {0}ms\nBatch size: {1}\nTraining dataset length: {2}\nTest dataset length: {3}\n", timer.ElapsedMilliseconds, dataloader.BatchSize, dataloader.TestSet.Length * BATCH_SIZE, dataloader.TrainSet.Length * BATCH_SIZE);
            return dataloader;
        }

        private static void TestNeuralNetwork(Stopwatch timer, DataLoader dataLoader)
        {
            Console.WriteLine("\n****** Commencing Neural Network Tests ******\n");
            Console.WriteLine("Creating Model");
            timer.Restart();
            var model = new Model();
            timer.Stop();
            Console.WriteLine("Model created in {0}ms", timer.ElapsedMilliseconds);
            model.setLog(true);
            Console.WriteLine("Running single pass with full logging");
            var batch = dataLoader.TrainSet[0];
            var x = batch.input;
            var y = batch.output;
            timer.Restart();
            var y_hat = model.Forward(x);
            timer.Stop();
            Console.WriteLine("Forward pass completed in {0}ms\n", timer.ElapsedMilliseconds);
            var MSELosses = LossFunctions.MSELoss(y_hat, y);
            var averageMSELoss = MSELosses.Sum() / BATCH_SIZE;
            var CELosses = LossFunctions.CrossEntropyLoss(y_hat, y);
            var averageCELoss = CELosses.Sum() / BATCH_SIZE;
            Console.WriteLine("Model predicted output:\n{0}\n\nActual output:\n{1}\n\nMean square error: {2}\nCross entropy loss: {3}\n", y_hat[0].ToString(), y[0].ToString(), averageMSELoss, averageCELoss);
        }
    }
}