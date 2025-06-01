import random
from collections import defaultdict
import matplotlib.pyplot as plt

class ViewpointSamplingSimulator:
    def __init__(self, n_viewpoints=20, n_iterations=1000):
        self.n_viewpoints = n_viewpoints
        self.n_iterations = n_iterations
        self.viewpoint_indices = list(range(n_viewpoints))
        self.selection_history = []
        self.current_cycle = []
        self.cycles_count = 0
    
    def reset_viewpoints(self):
        self.viewpoint_indices = list(range(self.n_viewpoints))
        self.cycles_count += 1
        print(f"\n새로운 사이클 #{self.cycles_count} 시작")
        print(f"사용 가능한 시점 수: {len(self.viewpoint_indices)}")
    
    def sample_viewpoint(self, iteration):
        if not self.viewpoint_indices:
            self.reset_viewpoints()
        
        rand_idx = random.randint(0, len(self.viewpoint_indices) - 1)
        selected_idx = self.viewpoint_indices.pop(rand_idx)
        
        self.selection_history.append((iteration, selected_idx))
        self.current_cycle.append(selected_idx)
        
        print(f"Iteration {iteration}: 선택된 시점 {selected_idx}, 남은 시점 수: {len(self.viewpoint_indices)}")
        return selected_idx
    
    def run_simulation(self):
        for i in range(self.n_iterations):
            self.sample_viewpoint(i)
    
    def analyze_results(self):
        # 1. 선택 빈도 분석
        selection_counts = defaultdict(int)
        for _, idx in self.selection_history:
            selection_counts[idx] += 1
        
        # 2. 시각화
        plt.figure(figsize=(15, 5))
        
        # 2.1 선택 빈도 히스토그램
        plt.subplot(1, 2, 1)
        plt.bar(selection_counts.keys(), selection_counts.values())
        plt.title('시점별 선택 빈도')
        plt.xlabel('시점 인덱스')
        plt.ylabel('선택된 횟수')
        
        # 2.2 선택 순서 시각화
        plt.subplot(1, 2, 2)
        iterations, viewpoints = zip(*self.selection_history)
        plt.scatter(iterations, viewpoints, alpha=0.5)
        plt.title('시점 선택 패턴')
        plt.xlabel('Iteration')
        plt.ylabel('선택된 시점')
        
        plt.tight_layout()
        plt.show()
        
        # 3. 통계 출력
        print("\n=== 시뮬레이션 결과 ===")
        print(f"총 사이클 수: {self.cycles_count}")
        print(f"평균 선택 횟수: {self.n_iterations / self.n_viewpoints:.2f}")
        print("\n시점별 선택 횟수:")
        for idx, count in sorted(selection_counts.items()):
            print(f"시점 {idx}: {count}회 선택됨")

# 시뮬레이션 실행
def run_test():
    # 테스트 파라미터 설정
    n_viewpoints = 20  # 시점 수
    n_iterations = 1000  # 총 반복 횟수
    
    print(f"=== 비복원추출 시뮬레이션 시작 ===")
    print(f"시점 수: {n_viewpoints}")
    print(f"총 반복 횟수: {n_iterations}\n")
    
    # 시뮬레이터 생성 및 실행
    simulator = ViewpointSamplingSimulator(n_viewpoints, n_iterations)
    simulator.run_simulation()
    simulator.analyze_results()

if __name__ == "__main__":
    run_test()