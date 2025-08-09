package lyz.okgct.backend.service.impl.transform;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import lyz.okgct.backend.dao.Transform;
import lyz.okgct.backend.mapper.TransformMapper;
import lyz.okgct.backend.service.impl.utils.FileProcessorUtil;
import lyz.okgct.backend.service.transform.TransformService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestParam;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class TransformServiceImpl implements TransformService {

    @Autowired
    private TransformMapper transformMapper;

    @Override
    public Transform getPage(@RequestParam Map<String, String> data) {
        return transformMapper.selectById(data.get("page_id"));
    }

    @Override
    public Transform addPage(@RequestParam Map<String, String> data) {
        Transform transform = new Transform();
        transform.setUserId(Integer.valueOf(data.get("user_id")));
        transform.setSourceCode("Please input your code");
        transform.setTargetCode("Waiting...");
        transformMapper.insert(transform);

        List<Transform> transforms = transformMapper.selectList(null);
        return transforms.get(transforms.size() - 1);
    }

    @Override
    public List<Transform> loadPage(@RequestParam Map<String, String> data) {
        QueryWrapper<Transform> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("user_id", data.get("user_id"));
        List<Transform> transforms = transformMapper.selectList(queryWrapper);

        if (transforms.isEmpty()) {
            Transform transform = new Transform();
            transform.setUserId(Integer.valueOf(data.get("user_id")));
            transform.setSourceCode("Please input your code");
            transform.setTargetCode("Waiting...");;
            transformMapper.insert(transform);
        }

        return transformMapper.selectList(queryWrapper);
    }

    @Override
    public Map<String, String> editPage(Map<String, String> data) {
        Transform transform = transformMapper.selectById(data.get("page_id"));
        transform.setTitle(data.get("title"));

        transformMapper.updateById(transform);

        Map<String, String> map = new HashMap<>();
        map.put("error_message", "success");
        return map;
    }

    public Map<String, String> closePage(Map<String, String> data) {
        transformMapper.deleteById(data.get("page_id"));

        Map<String, String> map = new HashMap<>();
        map.put("error_message", "success");
        return map;
    }

    @Override
    public Map<String, String> savePage(Map<String, String> data) {
        Transform transform = transformMapper.selectById(data.get("page_id"));

        transform.setSourceFramework(data.get("source_framework"));
        transform.setSourceVersion(data.get("source_version"));
        transform.setTargetFramework(data.get("target_framework"));
        transform.setTargetVersion(data.get("target_version"));
        transform.setCodeFrom(data.get("code_from"));
        transform.setModels(data.get("models"));

        transform.setSourceCode(data.get("source_code"));
        transform.setTargetCode(data.get("target_code"));
        transformMapper.updateById(transform);

        Map<String, String> map = new HashMap<>();
        map.put("error_message", "success");
        return map;
    }

    @Override
    public Transform changePage(Map<String, String> data) {
        return transformMapper.selectById(data.get("page_id"));
    }

    public String getTargetCodeByFile(String sourceFramework, String targetFramework, String sourceFilePath, String targetFilePath, String sourceTreePath, String targetTreePath) {
        String targetCode = "";

        try {
            String[] cmd = new String[8];
            cmd[0] = "D:/Anaconda/envs/astorchms/python.exe";  // Python解释器路径
            cmd[1] = "D:/Program/okgct/test/transform_file.py";  // 迁移脚本
            cmd[2] = sourceFramework;
            cmd[3] = targetFramework;
            cmd[4] = sourceFilePath;
            cmd[5] = targetFilePath;
            cmd[6] = sourceTreePath;
            cmd[7] = targetTreePath;

            ProcessBuilder pb = new ProcessBuilder(cmd);
            Process p = pb.start(); // 启动进程
            BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String ret = in.readLine(); // 读取输出
            int exitVal = p.waitFor(); // 等待进程结束并获取退出值

            targetCode = FileProcessorUtil.ReadFileToString(cmd[5]);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return targetCode;
    }

    @Override
    public Map<String, String> getTargetCodeByProject(@RequestParam Map<String, String> data) {
        String targetCode = "";

        Map<String, String> map = new HashMap<>();

        try {
            String[] cmd = new String[6];
            cmd[0] = "D:/Anaconda/envs/astorchms/python.exe";  // Python解释器路径
            cmd[1] = "D:/Program/okgct/test/transform_project.py";  // 迁移脚本
            cmd[2] = data.get("source_framework");
            cmd[3] = data.get("target_framework");
            cmd[4] = "D:/Program/okgct/datasets/backend/project";
            cmd[5] = "D:/Program/okgct/datasets/backend/project_new";

            ProcessBuilder pb = new ProcessBuilder(cmd);
            Process p = pb.start(); // 启动进程
            BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String ret = in.readLine(); // 读取输出
            int exitVal = p.waitFor(); // 等待进程结束并获取退出值

            targetCode = "Please check your local project";
            map.put("target_code", targetCode);
            map.put("source_tree", "null");
            map.put("target_tree", "null");
        } catch (Exception e) {
            e.printStackTrace();
        }

        return map;
    }

    @Override
    public Map<String, String> getTargetCodeWithInput(Map<String, String> data) {
        String targetCode = "";
        String sourceTree = "";
        String targetTree = "";

        try {
            String sourceFramework = data.get("source_framework");
            String targetFramework = data.get("target_framework");

            String sourceCode = data.get("source_code");
            String sourceFilePath = "D:/Program/okgct/datasets/backend/source_code.py";
            FileProcessorUtil.WriteStringToFile(sourceCode, sourceFilePath);

            String targetFilePath = "D:/Program/okgct/datasets/backend/source_code_new.py";

            String sourceTreePath = "D:/Program/okgct/datasets/backend/source_code_tree.txt";

            String targetTreePath = "D:/Program/okgct/datasets/backend/source_code_tree_new.txt";

            targetCode = getTargetCodeByFile(sourceFramework, targetFramework, sourceFilePath, targetFilePath, sourceTreePath, targetTreePath);
            sourceTree = FileProcessorUtil.ReadFileToString(sourceTreePath);
            targetTree = FileProcessorUtil.ReadFileToString(targetTreePath);

        } catch (Exception e) {
            e.printStackTrace();
        }

        Transform transform = transformMapper.selectById(data.get("page_id"));
        transform.setSourceTree(sourceTree);
        transform.setTargetTree(targetTree);
        transformMapper.updateById(transform);

        Map<String, String> map = new HashMap<>();
        map.put("target_code", targetCode);
        map.put("source_tree", sourceTree);
        map.put("target_tree", targetTree);
        return map;
    }

    @Override
    public Map<String, String> getTargetCodeWithUploadFile(@RequestParam Map<String, String> data) {
        String targetCode = "";
        String sourceTree = "";
        String targetTree = "";

        try {
            String sourceFramework = data.get("source_framework");
            String targetFramework = data.get("target_framework");

            String sourceFilePath = "D:/Program/okgct/datasets/backend/upload/source_code.py";
            String targetFilePath = "D:/Program/okgct/datasets/backend/upload/source_code_new.py";

            String sourceTreePath = "D:/Program/okgct/datasets/backend/upload/source_code_tree.py";
            String targetTreePath = "D:/Program/okgct/datasets/backend/upload/source_code_tree_new.py";


            targetCode = getTargetCodeByFile(sourceFramework, targetFramework, sourceFilePath, targetFilePath, sourceTreePath, targetTreePath);
            sourceTree = FileProcessorUtil.ReadFileToString(sourceTreePath);
            targetTree = FileProcessorUtil.ReadFileToString(targetTreePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Transform transform = transformMapper.selectById(data.get("page_id"));
        transform.setSourceTree(sourceTree);
        transform.setTargetTree(targetTree);
        transformMapper.updateById(transform);

        Map<String, String> map = new HashMap<>();
        map.put("target_code", targetCode);
        map.put("source_tree", sourceTree);
        map.put("target_tree", targetTree);
        return map;
    }
}
