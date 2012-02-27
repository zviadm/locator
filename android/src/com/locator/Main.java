package com.locator;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.view.View;
import android.widget.*;

import java.util.List;

public class Main extends Activity
{

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        final TextView textRouters = (TextView)findViewById(R.id.text_routers);

        IntentFilter intent = new IntentFilter();
        intent.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);

        registerReceiver(new BroadcastReceiver(){
            public void onReceive(Context context, Intent intent){
                // Code to execute when SCAN_RESULTS_AVAILABLE_ACTION event occurs
                WifiManager wm = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
                List<ScanResult> scanResults = wm.getScanResults(); // Returns a <list> of scanResults
                StringBuilder routers = new StringBuilder();
                for (ScanResult result : scanResults) {
                    routers.append("SSID: " + result.SSID + "\n");
                    routers.append("BSSID: " + result.BSSID + "\n");
                    routers.append("LEVEL: " + result.level + "\n");
                    routers.append("\n");
                }

                textRouters.setText(routers.toString());
            }
        }, intent );

        View butRefresh = findViewById(R.id.but_refresh);
        butRefresh.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Now you can call this and it should execute the broadcastReceiver's onReceive()
                WifiManager wm = (WifiManager) getSystemService(Context.WIFI_SERVICE);
                if (wm.getWifiState() == WifiManager.WIFI_STATE_ENABLED) {
                    wm.startScan();
                }
            }
        });
    }

}
