package com.locator;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.text.Html;
import android.text.format.Time;
import android.util.Log;
import android.view.View;
import android.widget.*;
import org.apache.http.HttpHost;
import org.apache.http.HttpRequest;
import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.protocol.HTTP;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.util.*;

public class Main extends Activity
{
    
    public static final String TAG = "LOCATOR_MAIN";

    private String deviceId = null;
    private boolean isTracking = false;

    private View butSnapshot = null;
    private View butStartTracking = null;
    private View butEndTracking = null;
    private View butGatherSamples = null;

    private EditText textNumOfSamples = null;
    private EditText textLocationId = null;
    private TextView textRouters = null;
    
    private ProgressBar progressSamples = null;
    
    private HttpClient httpClient = null;

    protected class TrackLocationTask extends AsyncTask<Map<String, Integer>, Void, Void> {
        private String deviceId;
        
        public TrackLocationTask(String deviceId) {
            super();
            this.deviceId = deviceId;
        }
        
        @Override
        protected Void doInBackground(Map<String, Integer>... listRouterLevels) {
            HttpHost host = new HttpHost("locator.dropbox.com", 80);
            
            for (Map<String, Integer> routerLevels : listRouterLevels) {
                JSONObject rpcDataObj=new JSONObject();
                try {
                    rpcDataObj.put("method", "track_location");
                    rpcDataObj.put("device_id", deviceId);
                    rpcDataObj.put("timestamp", (double) System.currentTimeMillis() / 1000.0);
                    JSONObject routerLevelsObj = new JSONObject();
                    for (Map.Entry<String, Integer> routerLevel : routerLevels.entrySet()) {
                        routerLevelsObj.put(routerLevel.getKey(), routerLevel.getValue());
                    }
                    rpcDataObj.put("router_levels", routerLevelsObj);
                } catch (JSONException e) {
                    Log.d(TAG, "Failed to json RPC data", e);
                    return null;
                }
                
                String rpcData = rpcDataObj.toString();
                HttpPost request = new HttpPost("/rpc");
                try {
                    request.setEntity(new StringEntity(rpcData, HTTP.UTF_8));
                } catch (UnsupportedEncodingException e) {
                    Log.d(TAG, "WTF????.", e);
                    return null;
                }

                HttpResponse response;
                try {
                    response = httpClient.execute(host, request);
                    response.getEntity().consumeContent();
                    Log.d(TAG, "HttpResponse StatusLine: " + response.getStatusLine().toString());
                } catch (ClientProtocolException e) {
                    Log.d(TAG, "Failed to send sample...", e);
                } catch (IOException e) {
                    Log.d(TAG, "Failed to send sample...", e);
                }
            }
            return null;
        }
        
        @Override
        protected void onPostExecute(Void result) {
            WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
                wm.startScan();
            }
        }
    }

    protected class LocationSampleTask extends AsyncTask<List<ScanResult>, Void, Void> {
        private String deviceId;
        private String locationId;

        public LocationSampleTask(String deviceId, String locationId) {
            super();
            this.deviceId = deviceId;
            this.locationId = locationId;
        }

        @Override
        protected Void doInBackground(List<ScanResult>... listScanResults) {
            HttpHost host = new HttpHost("locator.dropbox.com", 80);

            for (List<ScanResult> scanResults : listScanResults) {
                JSONObject rpcDataObj=new JSONObject();
                try {
                    rpcDataObj.put("method", "location_sample");
                    rpcDataObj.put("device_id", deviceId);
                    rpcDataObj.put("location_id", locationId);
                    rpcDataObj.put("timestamp", (double) System.currentTimeMillis() / 1000.0);
                    JSONArray scanResultsObj = new JSONArray();
                    for (ScanResult scanResult : scanResults) {
                        scanResultsObj.put(scanResultToJson(scanResult));
                    }
                    rpcDataObj.put("scan_results", scanResultsObj);
                } catch (JSONException e) {
                    Log.d(TAG, "Failed to json RPC data", e);
                    return null;
                }

                String rpcData = rpcDataObj.toString();
                HttpPost request = new HttpPost("/rpc");
                try {
                    request.setEntity(new StringEntity(rpcData, HTTP.UTF_8));
                } catch (UnsupportedEncodingException e) {
                    Log.d(TAG, "WTF????.", e);
                    return null;
                }

                HttpResponse response;
                try {
                    response = httpClient.execute(host, request);
                    response.getEntity().consumeContent();
                    Log.d(TAG, "HttpResponse StatusLine: " + response.getStatusLine().toString());
                } catch (ClientProtocolException e) {
                    Log.d(TAG, "Failed to send sample...", e);
                } catch (IOException e) {
                    Log.d(TAG, "Failed to send sample...", e);
                }
            }
            return null;
        }

        @Override
        protected void onPostExecute(Void result) {
            WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
                wm.startScan();
            }
        }
    }

    private static final Map<String, String> BSSID_TO_ROUTER =
            Collections.unmodifiableMap(new HashMap<String, String>() {{
                put("00:0b:86:74:96:80", "AP-4-01");
                put("00:0b:86:74:96:81", "AP-4-01");

                put("00:0b:86:74:97:60", "AP-4-02");
                put("00:0b:86:74:97:61", "AP-4-02");

                put("00:0b:86:74:9a:80", "AP-4-03");
                put("00:0b:86:74:9a:81", "AP-4-03");

                put("00:0b:86:74:9a:90", "AP-4-04");
                put("00:0b:86:74:9a:91", "AP-4-04");

                put("00:0b:86:74:97:90", "AP-4-05");
                put("00:0b:86:74:97:91", "AP-4-05");
            }});

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        httpClient = new DefaultHttpClient();

        butSnapshot = findViewById(R.id.but_snapshot);
        butStartTracking = findViewById(R.id.but_start_tracking);
        butEndTracking = findViewById(R.id.but_end_tracking);
        butGatherSamples = findViewById(R.id.but_gather_samples);
        progressSamples = (ProgressBar)findViewById(R.id.progress_samples);
        
        textNumOfSamples = (EditText)findViewById(R.id.text_num_samples);
        textLocationId = (EditText)findViewById(R.id.text_location_id);
        textRouters = (TextView)findViewById(R.id.text_routers);

        butSnapshot.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                WifiManager wm = (WifiManager) getSystemService(Context.WIFI_SERVICE);
                if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
                    disableScanningButtons();
                    getSnapshot(Integer.valueOf(textNumOfSamples.getText().toString()));
                } else {
                    Toast toast = Toast.makeText(getApplicationContext(), "wifi is turned off!", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
        });

        butStartTracking.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                WifiManager wm = (WifiManager) getSystemService(Context.WIFI_SERVICE);
                if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
                    if (deviceId == null) {
                        WifiManager wifiMan = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                        WifiInfo wifiInf = wifiMan.getConnectionInfo();
                        deviceId = wifiInf.getMacAddress();
                    }

                    butEndTracking.setEnabled(true);
                    isTracking = true;
                    disableScanningButtons();
                    startTracking(Integer.valueOf(textNumOfSamples.getText().toString()));
                } else {
                    Toast toast = Toast.makeText(getApplicationContext(), "wifi is turned off!", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
        });

        butEndTracking.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                isTracking = false;
                butEndTracking.setEnabled(false);
            }
        });

        butGatherSamples.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                WifiManager wm = (WifiManager) getSystemService(Context.WIFI_SERVICE);
                if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
                    if (deviceId == null) {
                        WifiManager wifiMan = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                        WifiInfo wifiInf = wifiMan.getConnectionInfo();
                        deviceId = wifiInf.getMacAddress();
                    }

                    disableScanningButtons();
                    gatherSamples(textLocationId.getText().toString(), Integer.valueOf(textNumOfSamples.getText().toString()));
                } else {
                    Toast toast = Toast.makeText(getApplicationContext(), "wifi is turned off!", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
        });
    }

    private JSONObject scanResultToJson(ScanResult scanResult) {
        JSONObject scanResultObj=new JSONObject();
        try {
            scanResultObj.put("SSID", scanResult.SSID);
            scanResultObj.put("BSSID", scanResult.BSSID);
            scanResultObj.put("level", scanResult.level);
            scanResultObj.put("frequency", scanResult.frequency);
            return scanResultObj;
        } catch (JSONException e) {
            Log.d(TAG, "Failed to json scanResult", e);
            return null;
        }
    }
    
    private Map<String, Integer> getRouterLevels(List<List<ScanResult>> listScanResults) {
        Map<String, Integer> routerLevels = new HashMap<String, Integer>();

        for (List<ScanResult> scanResults : listScanResults) {
            for (ScanResult scanResult : scanResults) {
                if (BSSID_TO_ROUTER.containsKey(scanResult.BSSID)) {
                    String router = BSSID_TO_ROUTER.get(scanResult.BSSID);
                    if (routerLevels.containsKey(router)) {
                        routerLevels.put(router, Math.max(scanResult.level, routerLevels.get(router)));
                    } else {
                        routerLevels.put(router, scanResult.level);
                    }
                }
            }
        }
        return routerLevels;        
    }
    
    private void setRoutersInfo(Map<String, Integer> routerLevels, List<ScanResult> scanResults) {
        StringBuilder routersInfo = new StringBuilder();
        routersInfo.append("Last Updated: " + Calendar.getInstance().getTime().toString() + "\n\n");
        for (Map.Entry<String, Integer> routerLevel : routerLevels.entrySet()) {
            routersInfo.append("Router: " + routerLevel.getKey() + "\n");
            routersInfo.append("LEVEL: " + routerLevel.getValue().toString() + "\n");
            routersInfo.append("\n");
        }
        routersInfo.append("\n");
        Collections.sort(scanResults, new Comparator<ScanResult>() {
            @Override
            public int compare(ScanResult scanResult, ScanResult scanResult1) {
                return scanResult1.level - scanResult.level;
            }
        });
        for (ScanResult scanResult : scanResults) {
            routersInfo.append("BSSID: " + scanResult.BSSID + "\n");
            routersInfo.append("SSID: "  + scanResult.SSID + "\n");
            routersInfo.append("LEVEL: " + scanResult.level + "\n");
            routersInfo.append("\n");
        }
        textRouters.setText(routersInfo.toString());
    }

    private void getSnapshot(final int numSamples) {
        IntentFilter intent = new IntentFilter();
        intent.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);
        registerReceiver(new BroadcastReceiver(){
            int samplesLeft = numSamples;
            List<List<ScanResult>> listScanResults = new ArrayList<List<ScanResult>>();
            
            public void onReceive(Context context, Intent intent){
                WifiManager wm = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
                List<ScanResult> scanResults = wm.getScanResults();
                listScanResults.add(scanResults);
                samplesLeft -= 1;
                
                if (samplesLeft == 0) {
                    Map<String, Integer> routerLevels = getRouterLevels(listScanResults);
                    setRoutersInfo(routerLevels, scanResults);

                    enableScanningButtons();
                    unregisterReceiver(this);
                } else {
                    if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
                        wm.startScan();
                    }
                }
            }
        }, intent );

        WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
            wm.startScan();
        }
    }

    private void startTracking(final int numSamples) {
        IntentFilter intent = new IntentFilter();
        intent.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);
        registerReceiver(new BroadcastReceiver(){
            List<List<ScanResult>> listScanResults = new LinkedList<List<ScanResult>>();

            public void onReceive(Context context, Intent intent){
                WifiManager wm = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
                List<ScanResult> scanResults = wm.getScanResults();
                listScanResults.add(scanResults);

                if (!isTracking) {
                    enableScanningButtons();
                    unregisterReceiver(this);
                } else {
                    if (listScanResults.size() > numSamples) {
                        listScanResults.remove(0);
                        Map<String, Integer> routerLevels = getRouterLevels(listScanResults);
                        setRoutersInfo(routerLevels, scanResults);

                        TrackLocationTask task = new TrackLocationTask(deviceId);
                        Map[] tmpRouterLevels = {routerLevels, };
                        task.execute(tmpRouterLevels);
                    } else {
                        if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
                            wm.startScan();
                        }
                    }
                }
            }
        }, intent );

        WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
            wm.startScan();
        }
    }

    private void gatherSamples(final String locationId, final int numSamples) {
        progressSamples.setMax(numSamples);
        progressSamples.setProgress(0);

        IntentFilter intent = new IntentFilter();
        intent.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);
        registerReceiver(new BroadcastReceiver(){

            private int samplesLeft = numSamples;
            
            public void onReceive(Context context, Intent intent){
                WifiManager wm = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
                List<ScanResult> scanResults = wm.getScanResults();
                samplesLeft -= 1;
                progressSamples.setProgress(numSamples - samplesLeft);

                LocationSampleTask task = new LocationSampleTask(deviceId, locationId);
                List[] tmpScanResults = {scanResults, };
                task.execute(tmpScanResults);

                if (samplesLeft == 0) {
                    enableScanningButtons();
                    unregisterReceiver(this);
                }
            }
        }, intent );

        WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
            wm.startScan();
        }
    }

    private void disableScanningButtons() {
        butSnapshot.setEnabled(false);
        butStartTracking.setEnabled(false);
        butGatherSamples.setEnabled(false);
    }

    private void enableScanningButtons() {
        butSnapshot.setEnabled(true);
        butStartTracking.setEnabled(true);
        butGatherSamples.setEnabled(true);
    }

    @Override
    public void onPause() {
        super.onPause();
        isTracking = false;
    }
}
