from node import Node
from collections import deque
from queue import Queue
import heapq

class Search:
    @staticmethod
    def BFS(s):
        open_queue = deque()
        closed_list = []
        step = 0
        init_node = Node(s, None, None)
        if s.isGoal():
            return init_node , 0

        open_queue.append(init_node)
        
        while open_queue:
            current = open_queue.popleft()  # Dequeue the shallowest node in Open
            closed_list.append(current)
            step +=1
            for (action, successor) in current.state.successorFunction():
                child = Node(successor, current, action)
                if step==2 or step==3 or step==4 or step==5 or step==6:
                    print(f'step :{step}action : {child.action}h :{child.f}')
                
                if child.state.board not in [node.state.board for node in closed_list] and \
                    not any(node.state.board == child.state.board for node in open_queue):
                    if child.state.isGoal():
                        return child , step
                    open_queue.append(child)
        
        return None
    @staticmethod
    def a_star(s,hewristic):
        open = []
        closed = set()
        init_node = Node(s, None, None,1,hewristic)
        step = 0
        heapq.heappush(open, (init_node.f, init_node))

        while open:
            current = heapq.heappop(open)[1]
            if current.state.isGoal():
                return current , step

            step+=1

            closed.add(current)
            print(step)
            for action, successor in current.state.successorFunction():
                child = Node(successor, current, action,1,hewristic)

                if child.state.board not in [node.state.board for node in closed]:
                    if child.state.board not in [node.state.board for f , node in open]:
                        heapq.heappush(open, (child.f, child))
                    else:
                        for f, node in open:
                            if node.state.board == child.state.board and f > child.f:
                                open.remove((f, node))
                                heapq.heappush(open, (child.f, child))
                                break
                else:
                    for node in closed:
                        if node.state.board == child.state.board and node.f > child.f:
                            closed.remove(node)
                            heapq.heappush(open, (child.f, child))
                            break
        return None , 0
