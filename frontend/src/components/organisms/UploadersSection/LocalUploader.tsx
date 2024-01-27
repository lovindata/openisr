import { useBackend } from "../../../services/backend";
import { paths } from "../../../services/backend/endpoints";
import { BorderBox } from "../../atoms/BorderBox";
import { SvgIcon } from "../../atoms/SvgIcon";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";

export function LocalUploader() {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: uploadImages } = useMutation({
    mutationFn: async (files: File[]) => {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) formData.append("files", files[i]);
      return backend
        .post<
          paths["/images/upload-local"]["post"]["responses"]["200"]["content"]["application/json"]
        >(`/images/upload-local`, formData)
        .then((_) => _.data);
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["/images"] }),
  });

  const { getRootProps, getInputProps } = useDropzone({
    accept: { "images/*": [] },
    onDrop: (acceptedFiles) => uploadImages(acceptedFiles),
  });

  return (
    <BorderBox dashed className="h-16 w-32">
      <div {...getRootProps()} className="h-full">
        <input {...getInputProps()} />
        <div className="mx-auto flex h-full w-min items-center space-x-3">
          <SvgIcon type="folder" className="h-8 w-8" />
          <label>Local upload</label>
        </div>
      </div>
    </BorderBox>
  );
}
