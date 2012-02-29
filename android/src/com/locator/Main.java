package com.locator;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.http.AndroidHttpClient;
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
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.protocol.HTTP;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.List;

public class Main extends Activity
{
    
    public static final String TAG = "LOCATOR_MAIN";

    private int samplesToSend = 0;
    private String deviceId = null;
    
    private View butSendSamples = null;
    private EditText textNumOfSamples = null;
    private EditText textLocationId = null;
    private ProgressBar progressSamples = null;

    private class SendSampleTask extends AsyncTask<String, Void, String> {
        private int samplesSent = 0;
        
        protected String doInBackground(String... sampleJsons) {
            HttpClient httpClient = AndroidHttpClient.newInstance("Locator 1.0/Android", getApplicationContext());
            HttpHost host = new HttpHost("locator.dropbox.com", 80);
            for (String sampleJson : sampleJsons) {
                HttpPost request = new HttpPost("/rpc");
                try {
                    request.setEntity(new StringEntity(sampleJson, HTTP.UTF_8));
                } catch (UnsupportedEncodingException e) {
                    Log.d(TAG, "WTF????.", e);
                }

                HttpResponse response = null;
                Exception exc = null;
                try {
                    response = httpClient.execute(host, request);
                    Log.d(TAG, "HttpResponse StatusLine: " + response.getStatusLine().toString());
                } catch (ClientProtocolException e) {
                    Log.d(TAG, "Failed to send sample...", e);
                    exc = e;
                } catch (IOException e) {
                    Log.d(TAG, "Failed to send sample...", e);
                    exc = e;
                }
                
                if (response == null || response.getStatusLine().getStatusCode() != 200) {
                    String errMsg;
                    if (exc != null) {
                        errMsg = "Failed to send a sample: " + exc.getMessage();
                    } else if (response != null) {
                        errMsg = "Failed to send a sample: " + response.getStatusLine().toString();
                    } else {
                        errMsg = "WTF??";
                    }
                    return errMsg;
                }
                samplesSent += 1;
            }
            return null;
        }

        protected void onPostExecute(String errMsg) {
            progressSamples.setProgress(progressSamples.getProgress() + samplesSent);
            samplesToSend -= samplesSent;

            if (errMsg != null) {
                Toast toast = Toast.makeText(getApplicationContext(), errMsg, Toast.LENGTH_SHORT);
                toast.show();
            } else {
                if (samplesToSend > 0) {
                    WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                    wm.startScan();
                    return;
                }
            }

            if (samplesToSend == 0) {
                textLocationId.setText("");
            } else {
                // If some sort of error happened stop sending samples and quit
                textNumOfSamples.setText(Integer.toString(samplesToSend));
                samplesToSend = 0;
            }
            butSendSamples.setEnabled(true);
            textLocationId.setEnabled(true);
            textNumOfSamples.setEnabled(true);
        }
    }
    
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        butSendSamples = findViewById(R.id.but_send_samples);
        textNumOfSamples = (EditText)findViewById(R.id.text_num_samples);
        textLocationId = (EditText)findViewById(R.id.text_location_id);
        progressSamples = (ProgressBar)findViewById(R.id.progress_samples);
        final TextView textRouters = (TextView)findViewById(R.id.text_routers);

        IntentFilter intent = new IntentFilter();
        intent.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);
        registerReceiver(new BroadcastReceiver(){
            public void onReceive(Context context, Intent intent){
                WifiManager wm = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
                List<ScanResult> scanResults = wm.getScanResults(); // Returns a <list> of scanResults
                // set routers view too
                StringBuilder routers = new StringBuilder();
                for (ScanResult result : scanResults) {
                    if (result.SSID.toLowerCase().startsWith("dropbox")) {
                        routers.append("SSID: " + result.SSID + "\n");
                        routers.append("BSSID: " + result.BSSID + "\n");
                        routers.append("LEVEL: " + result.level + "\n");
                        routers.append("\n");
                    }
                }
                textRouters.setText(routers.toString());

                if (samplesToSend == 0) {
                    return;
                }
                JSONObject sampleObj=new JSONObject();
                try {
                    sampleObj.put("method", "location_sample");
                    sampleObj.put("device_id", deviceId);
                    sampleObj.put("location_id", textLocationId.getText().toString());
                    sampleObj.put("timestamp", (double) System.currentTimeMillis() / 1000.0);
                    //obj.put("location_id", deviceId);

                    JSONArray scanResultList=new JSONArray();
                    for (ScanResult scanResult : scanResults) {
                        JSONObject scanResultObj=new JSONObject();
                        scanResultObj.put("SSID", scanResult.SSID);
                        scanResultObj.put("BSSID", scanResult.BSSID);
                        scanResultObj.put("level", scanResult.level);
                        scanResultObj.put("frequency", scanResult.frequency);
                        scanResultList.put(scanResultObj);
                    }
                    sampleObj.put("scan_results", scanResultList);
                } catch (JSONException e) {
                    Log.d(TAG, "Failed to json...", e);
                }
                String[] sampleJson = {sampleObj.toString()};
                AsyncTask sendSampleTask = new SendSampleTask();
                sendSampleTask.execute(sampleJson);
            }
        }, intent );

        butSendSamples.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                WifiManager wm = (WifiManager) getSystemService(Context.WIFI_SERVICE);
                if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
                    if (textLocationId.getText().toString().equals("")) {
                        Toast toast = Toast.makeText(getApplicationContext(), "need to specify locationId!", Toast.LENGTH_SHORT);
                        toast.show();

                        samplesToSend = 0;
                    } else {
                        butSendSamples.setEnabled(false);
                        textLocationId.setEnabled(false);
                        textNumOfSamples.setEnabled(false);

                        if (deviceId == null) {
                            WifiManager wifiMan = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                            WifiInfo wifiInf = wifiMan.getConnectionInfo();
                            deviceId = wifiInf.getMacAddress();
                        }

                        samplesToSend = Integer.parseInt(textNumOfSamples.getText().toString());
                        progressSamples.setMax(samplesToSend);
                        progressSamples.setProgress(0);
                    }

                    wm.startScan();
                } else {
                    Toast toast = Toast.makeText(getApplicationContext(), "wifi is turned off!", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
        });
    }

}
