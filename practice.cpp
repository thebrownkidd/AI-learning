#include <iostream>
#include <vector>
using namespace std;
int adjacent(int current, int* from, int* to,int iter = 0, int len){
    int* adj;
    iter += 1;
    int* nodes_comp[len] = { 0 }
    for(int i = 0; i< sizeof(from);i++){

    }
    for(int i = 0; i< sizeof(adj);i++){
        if(nodes_comp[adj[i]-1] < 1){
            nodes_comp[adj[i]-1] = 1;
            
        }
    }
    
}
int secondsToDisarmNetwork(int target, int network_nodes,int* network_from,int* network_to) {
    bool t = true;
    int comp_attacked = 0;
    int nodes_comp[network_nodes-1] = { 0 };
    int current_node = target;
    // while(comp_attacked < network_nodes){
    //     int* a = adjacent(target,network_from,network_to);
    //     comp_attacked += sizeof(a);
    //     while(t){
    //         for(int i = 0; i<sizeof(a);i++){
    //             if (nodes_comp[a[i]] < 1){
    //                 nodes_comp[a[i]] = 1;

    //             }
    //         }
    //     }
    // }
}

