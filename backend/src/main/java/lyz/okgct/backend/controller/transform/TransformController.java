package lyz.okgct.backend.controller.transform;

import lyz.okgct.backend.dao.Transform;
import lyz.okgct.backend.service.transform.TransformService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@RestController
@RequestMapping("transform/")
public class TransformController {

    @Autowired
    TransformService transformService;

    @PostMapping("page/info/")
    public Transform getPageInfo(@RequestParam Map<String, String> data) {
        return transformService.getPage(data);
    }

    @PostMapping("page/add/")
    public Transform addPage(@RequestParam Map<String, String> data) {
        return transformService.addPage(data);
    }

    @PostMapping("page/list/")
    public List<Transform> loadPage(@RequestParam Map<String, String> data) {
        return transformService.loadPage(data);
    }

    @PostMapping("page/edit/")
    public Map<String, String> editPage(@RequestParam Map<String, String> data) {
        return transformService.editPage(data);
    }

    @PostMapping("page/close/")
    public Map<String, String> closePage(@RequestParam Map<String, String> data) {
        return transformService.closePage(data);
    }

    @PostMapping("upload/")
    public Map<String, String> uploadFile(@RequestParam("file") MultipartFile file, @RequestParam Map<String, String> data) {
        try {
            if (Objects.requireNonNull(file.getOriginalFilename()).endsWith("py")) {
                file.transferTo(new File("D:/Program/okgct/datasets/backend/upload/source_code.py"));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return null;
    }

    @PostMapping("page/save/")
    public Map<String, String> savePage(@RequestParam Map<String, String> data) {
        return transformService.savePage(data);
    }

    @PostMapping("page/change/")
    public Map<String, String> changePage(@RequestParam Map<String, String> data) {
        Map<String, String> res = new HashMap<>();
        Transform transform = transformService.changePage(data);

        res.put("source_framework", transform.getSourceFramework());
        res.put("source_version", transform.getSourceVersion());
        res.put("target_framework", transform.getTargetFramework());
        res.put("target_version", transform.getTargetVersion());
        res.put("code_from", transform.getCodeFrom());
        res.put("models", transform.getModels());
        res.put("source_code", transform.getSourceCode());
        res.put("target_code", transform.getTargetCode());

        return res;
    }

    @PostMapping("target/input/")
    public Map<String, String> getTargetCodeWithInput(@RequestParam Map<String, String> data) {
        return transformService.getTargetCodeWithInput(data);
    }

    @PostMapping("target/upload/file/")
    public Map<String, String> getTargetCodeWithUploadFile(@RequestParam Map<String, String> data) {
        return transformService.getTargetCodeWithUploadFile(data);
    }

    @PostMapping("target/project/")
    public Map<String, String> getTargetCodeWithProject(@RequestParam Map<String, String> data) {
        return transformService.getTargetCodeByProject(data);
    }
}
