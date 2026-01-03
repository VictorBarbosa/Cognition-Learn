using System;
using System.Text;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Side channel to receive visual monitor metadata from Python (Algorithm, Port, Step).
    /// </summary>
    public class VisualMonitorSideChannel : SideChannel
    {
        public string Algorithm { get; private set; } = "Waiting...";
        public int Port { get; private set; } = 0;
        public int Step { get; private set; } = 0;
        public string CheckpointPath { get; private set; } = "";

        // Event to notify Agent when data changes
        public event Action OnMetadataUpdated;

        public VisualMonitorSideChannel()
        {
            // UUID must match the Python side
            ChannelId = new Guid("534c891e-810f-11ef-8a6a-325096b39f47");
            Debug.Log($"[VisualMonitorSideChannel] Initialized with ID: {ChannelId}");
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            try
            {
                // Protocol:
                // 1. Algorithm Name (String)
                // 2. Port (Int32)
                // 3. Step (Int32)
                // 4. Checkpoint Path (String)

                Algorithm = msg.ReadString();
                Port = msg.ReadInt32();
                Step = msg.ReadInt32();
                CheckpointPath = msg.ReadString();

                if (OnMetadataUpdated != null)
                {
                    OnMetadataUpdated.Invoke();
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[VisualMonitorSideChannel] Error reading message: {e.Message}");
            }
        }
    }
}
