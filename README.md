# Logistics Distribution Route Optimization

## Introduction  
This project focuses on optimizing logistics distribution routes using different algorithms to determine the most efficient path. The goal is to analyze various approaches to the **Traveling Salesman Problem (TSP)** and compare their effectiveness in minimizing travel costs.

- **Project Title:** Logistics Distribution Route Optimization  
- **Advisor:** Professor Chi-Hua Yu  

---

## Design Concept  
One day, while cycling through Tainan, I suddenly wondered how truck drivers find the optimal route to logistics centers despite the chaotic traffic. This curiosity led me to explore the possibility of creating a program similar to Google Maps that can recommend the shortest route.

---

## Implementation Details  
### **Technologies Used**  
- **Programming Language:** Python  
- **User Interface Framework:** Flet for GUI development  

### **Design Process**  
1. **User Input Handling:**  
   - Allow users to input the number of destinations, the starting point, and select an algorithm.  
2. **Algorithm Implementation:**  
   - Develop six commonly used algorithms to solve the **TSP (Traveling Salesman Problem)**.  
3. **Visualization:**  
   - Design a GUI interface that visually presents the routes taken by each algorithm.  

---

## Numerical Methods Used  
### **Optimization in TSP**  
Optimization, in simple terms, is the process of finding the **extremum** of an equation.  
For TSP, the goal is to find the **shortest route** while considering traffic conditions and distances as weights between locations.

- **Genetic Algorithm (GA):**  
  - Randomly generates an initial population of routes.  
  - Uses crossover and mutation operations to optimize paths.  
- **Ant Colony Optimization (ACO):**  
  - Simulates ants moving through a graph, selecting paths based on pheromone concentration and visibility.  
  - Adjusts pheromone levels iteratively to optimize routes.  

By applying different algorithms, we analyze which method achieves the shortest path with the least cost (weight) while ensuring all destinations are visited and returning to the starting point.

---

## Features & Functionalities  
### **Current Capabilities**  
- **Comparison of Six TSP Algorithms**: Evaluate different methods to determine the best path.  
- **Interactive GUI**: Users can input parameters and visualize results dynamically.  

### **Future Improvements**  
There are several enhancements planned for this project:  
1. **Automated Best Solution Selection**  
   - The system will automatically determine the optimal route based on user input and display only the best solution animation.  
2. **Custom Graph Input**  
   - Users will be able to input a custom graph, including nodes and weighted edges.  
   - If the user omits weights, the system will assign a large default value to prevent errors.  
3. **Enhanced GUI and 3D Visualization**  
   - Improve the graphical interface and implement 3D animations for route visualization.  
4. **Traffic Considerations & Multi-Vehicle Optimization**  
   - Incorporate real-time traffic conditions and multi-vehicle route planning for further optimization.  

---

## Results & Demonstration  
**GUI will look like this**
![image](https://github.com/yensha/Numerical_Final/blob/main/Test/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202025-02-08%20000809.png?raw=true))
A demonstration of the project can be found on YouTube:  
[![YouTube Video](https://img.youtube.com/vi/xWOcHnAKNi4/0.jpg)](https://youtu.be/xWOcHnAKNi4)  
🔗 **[Watch the video here](https://youtu.be/xWOcHnAKNi4)**  

---

## Conclusion  
This project successfully demonstrates various approaches to solving the **TSP** using different algorithms. The findings highlight the strengths and weaknesses of each method and pave the way for future improvements, particularly in **real-world logistics applications**.

