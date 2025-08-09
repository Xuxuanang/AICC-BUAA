package lyz.okgct.backend.service.transform;

import lyz.okgct.backend.dao.Transform;
import org.springframework.web.bind.annotation.RequestParam;
import java.util.List;
import java.util.Map;

public interface TransformService {

    Transform getPage(@RequestParam Map<String, String> data);

    Transform addPage(@RequestParam Map<String, String> data);

    List<Transform> loadPage(@RequestParam Map<String, String> data);

    Map<String, String> editPage(Map<String, String> data);

    Map<String, String> closePage(Map<String, String> data);

    Map<String, String> savePage(@RequestParam Map<String, String> data);

    Transform changePage(@RequestParam Map<String, String> data);

    Map<String, String> getTargetCodeWithInput(@RequestParam Map<String, String> data);

    Map<String, String> getTargetCodeWithUploadFile(@RequestParam Map<String, String> data);

    Map<String, String> getTargetCodeByProject(@RequestParam Map<String, String> data);
}
