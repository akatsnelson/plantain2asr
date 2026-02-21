from typing import List, Tuple, Literal

OpType = Literal["correct", "sub", "del", "ins"]

def align_words(ref: List[str], hyp: List[str]) -> List[Tuple[OpType, str, str]]:
    """
    Выполняет выравнивание двух последовательностей слов (Needleman-Wunsch algorithm).
    Возвращает список операций (type, ref_word, hyp_word).
    """
    n = len(ref)
    m = len(hyp)
    
    # dp[i][j] = (cost, op_type)
    dp = [[(0, "")] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = (i, "del")
    for j in range(m + 1):
        dp[0][j] = (j, "ins")
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                cost = 0
                op = "correct"
            else:
                cost = 1
                op = "sub"
                
            match_cost = dp[i-1][j-1][0] + cost
            del_cost = dp[i-1][j][0] + 1
            ins_cost = dp[i][j-1][0] + 1
            
            if match_cost <= del_cost and match_cost <= ins_cost:
                dp[i][j] = (match_cost, op)
            elif del_cost <= ins_cost:
                dp[i][j] = (del_cost, "del")
            else:
                dp[i][j] = (ins_cost, "ins")
                
    # Backtrace
    i, j = n, m
    alignment = []
    
    while i > 0 or j > 0:
        op = dp[i][j][1]
        
        if op == "correct" or op == "sub":
            alignment.append((op, ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1
        elif op == "del":
            alignment.append(("del", ref[i-1], "<eps>"))
            i -= 1
        elif op == "ins":
            alignment.append(("ins", "<eps>", hyp[j-1]))
            j -= 1
            
    return list(reversed(alignment))
