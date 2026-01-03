using System;
using System.IO;
using Unity.InferenceEngine;
using UnityEngine;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents
{
    /// <summary>
    /// Manager component for handling visual monitoring capabilities including
    /// best model visualization and side channel communication.
    /// Note: Direct ONNX runtime loading requires additional conversion infrastructure.
    /// </summary>
    [DefaultExecutionOrder(-100)] // Initialize before agents
    public class VisualMonitorManager : MonoBehaviour
    {
        // Reference to the visual monitor side channel
        private static Unity.MLAgents.SideChannels.VisualMonitorSideChannel s_VisualMonitorChannel;

        // Current inference device
        private InferenceDevice m_CurrentInferenceDevice = InferenceDevice.Default;

        // Flag to indicate if we're in visual monitoring mode
        private bool m_IsVisualMonitoringEnabled = false;

        // Cached references to agents that should use visual monitoring
        private Agent[] m_VisualMonitoringAgents;

        /// <summary>
        /// Initialize the visual monitor manager
        /// </summary>
        public void Initialize()
        {
            // The VisualMonitorSideChannel is already registered in Agent.cs
            // We don't need to register it again to avoid duplicate registration errors
            Debug.Log("[VisualMonitor] Visual Monitor Manager initialized");

            // Find all agents in the scene that should support visual monitoring
            // Use FindObjectsByType with no sorting for better performance
            m_VisualMonitoringAgents = FindObjectsByType<Agent>(FindObjectsSortMode.None);
        }

        /// <summary>
        /// Callback when new visual monitor metadata is received from Python
        /// </summary>
        private void OnVisualMonitorMetadataReceived()
        {
            if (s_VisualMonitorChannel == null) return;
            
            string checkpointPath = s_VisualMonitorChannel.CheckpointPath;
            
            if (string.IsNullOrEmpty(checkpointPath) || checkpointPath == "No Model Loaded" || checkpointPath == "None")
            {
                Debug.Log("[VisualMonitor] No model path received, continuing with current model");
                return;
            }
            
            // Check if this is an ONNX file path
            if (Path.GetExtension(checkpointPath).ToLower() == ".onnx")
            {
                if (File.Exists(checkpointPath))
                {
                    LoadOnnxModelAtRuntime(checkpointPath);
                }
                else
                {
                    Debug.LogWarning($"[VisualMonitor] ONNX file does not exist: {checkpointPath}");
                }
            }
        }

        /// <summary>
        /// Process a model file path received from Python for runtime model swapping
        /// In Unity, this involves using the model path to update agents via side channels
        /// </summary>
        /// <param name="modelPath">Path to the model file</param>
        public void LoadOnnxModelAtRuntime(string modelPath)
        {
            if (string.IsNullOrEmpty(modelPath))
            {
                Debug.LogError($"[VisualMonitor] Invalid model file path: {modelPath}");
                return;
            }

            try
            {
                Debug.Log($"[VisualMonitor] Model file received for runtime loading: {modelPath}");

                // Process the model path (this is where model swapping logic would happen)
                ProcessModelPath(modelPath);

                // Apply the model path to all agents
                ApplyModelPathToAgents(modelPath);

                m_IsVisualMonitoringEnabled = true;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[VisualMonitor] Error handling model path: {e.Message}");
            }
        }

        /// <summary>
        /// Process model path for runtime model swapping.
        /// In real implementation, this could be used to trigger pre-loaded model swapping.
        /// </summary>
        /// <param name="modelPath">Path to the model file</param>
        private void ProcessModelPath(string modelPath)
        {
            if (string.IsNullOrEmpty(modelPath))
            {
                Debug.LogError("[VisualMonitor] Model path is null or empty");
                return;
            }

            try
            {
                var extension = Path.GetExtension(modelPath).ToLower();

                if (extension == ".onnx" || extension == ".nn")
                {
                    // In a real implementation, you might have pre-loaded models in a dictionary/cache
                    // and switch to the appropriate one based on the received path
                    Debug.Log($"[VisualMonitor] Received model path for swapping: {modelPath}");

                    // This is where you'd implement the actual model swapping logic
                    // For example, you could have pre-loaded models identified by name/path
                    // and switch agents to use the appropriate model
                }
                else
                {
                    Debug.LogWarning($"[VisualMonitor] Unrecognized model format: {extension} for {modelPath}");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[VisualMonitor] Error processing model path {modelPath}: {e.Message}");
            }
        }

        /// <summary>
        /// Apply a model path to all agents in the scene
        /// This will set up the agents to use the new model for inference
        /// </summary>
        /// <param name="modelPath">Path to the model file</param>
        private void ApplyModelPathToAgents(string modelPath)
        {
            if (m_VisualMonitoringAgents == null) return;

            foreach (var agent in m_VisualMonitoringAgents)
            {
                if (agent != null)
                {
                    var behaviorParams = agent.GetComponent<BehaviorParameters>();
                    if (behaviorParams != null)
                    {
                        try
                        {
                            // For the dummy environment, we can use the SetModel method to swap the model
                            // This will force the agent to use the new model for inference
                            string behaviorName = behaviorParams.BehaviorName;

                            // In a real implementation, we might want to pre-load the model as a ModelAsset
                            // For now, we'll use the SetRuntimeModel method if available, or just trigger policy reload
                            Debug.Log($"[VisualMonitor] Agent {agent.name} received model path: {modelPath} - reloading policy");

                            // Force the agent to reload its policy to use any new model
                            agent.ReloadPolicy();
                        }
                        catch (System.Exception e)
                        {
                            Debug.LogError($"[VisualMonitor] Error processing model path for agent {agent.name}: {e.Message}");
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Apply the loaded model to all agents in the scene
        /// </summary>
        /// <param name="model">The loaded Sentis model</param>
        /// <param name="inferenceDevice">The inference device to use</param>
        private void ApplyModelToAgents(Model model, InferenceDevice inferenceDevice)
        {
            if (m_VisualMonitoringAgents == null) return;

            foreach (var agent in m_VisualMonitoringAgents)
            {
                if (agent != null)
                {
                    // Get the behavior parameters of the agent
                    var behaviorParams = agent.GetComponent<BehaviorParameters>();
                    if (behaviorParams != null)
                    {
                        try
                        {
                            // Set the runtime model for the agent using the BehaviorParameters method
                            behaviorParams.SetRuntimeModel(model);

                            // Reload the policy to use the new model
                            agent.ReloadPolicy();

                            Debug.Log($"[VisualMonitor] Applied model to agent: {agent.name}");
                        }
                        catch (System.Exception e)
                        {
                            Debug.LogError($"[VisualMonitor] Error applying model to agent {agent.name}: {e.Message}");
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Get the current visual monitor side channel
        /// </summary>
        public static Unity.MLAgents.SideChannels.VisualMonitorSideChannel GetVisualMonitorChannel()
        {
            return s_VisualMonitorChannel;
        }

        /// <summary>
        /// Check if visual monitoring is enabled
        /// </summary>
        public bool IsVisualMonitoringEnabled()
        {
            return m_IsVisualMonitoringEnabled;
        }

        private void OnDestroy()
        {
            // Clean up resources if needed
        }
    }
}