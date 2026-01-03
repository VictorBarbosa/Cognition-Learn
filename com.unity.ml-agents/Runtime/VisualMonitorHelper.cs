using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents
{
    /// <summary>
    /// Helper script to add visual monitor functionality to a scene
    /// </summary>
    [ExecuteInEditMode]
    public class VisualMonitorHelper : MonoBehaviour
    {
        [Tooltip("Enable this to activate visual monitoring for the agents in this scene")]
        public bool enableVisualMonitor = true;
        
        [Tooltip("When checked, this component will automatically find and assign agents in the scene")]
        public bool autoAssignAgents = true;
        
        void Start()
        {
            if (enableVisualMonitor)
            {
                SetupVisualMonitoring();
            }
        }
        
        void OnValidate()
        {
            if (enableVisualMonitor && autoAssignAgents)
            {
                SetupVisualMonitoring();
            }
        }
        
        public void SetupVisualMonitoring()
        {
            // Create or get the VisualMonitorManager
            var visualMonitorManager = UnityEngine.Object.FindFirstObjectByType<VisualMonitorManager>();
            if (visualMonitorManager == null)
            {
                var managerObj = new GameObject("VisualMonitorManager");
                visualMonitorManager = managerObj.AddComponent<VisualMonitorManager>();
            }

            visualMonitorManager.Initialize();
        }
    }
    
#if UNITY_EDITOR
    [CustomEditor(typeof(VisualMonitorHelper))]
    public class VisualMonitorHelperEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            
            VisualMonitorHelper script = (VisualMonitorHelper)target;
            
            if (GUILayout.Button("Setup Visual Monitoring"))
            {
                script.SetupVisualMonitoring();
            }
            
            if (GUILayout.Button("Find All Agents"))
            {
                var agents = FindObjectsOfType<Agent>();
                Debug.Log($"Found {agents.Length} agents in scene");
            }
        }
    }
#endif
}