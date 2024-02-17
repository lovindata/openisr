import { useBackend } from "../../../services/backend";
import { components, paths } from "../../../services/backend/endpoints";
import { Button } from "../../molecules/Button";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface Props {
  error: string;
  image: components["schemas"]["ImageODto"];
  onSuccessSubmit?: () => void;
}

export function ErrorContents({ error, image, onSuccessSubmit }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: retryLatestProcess, isPending } = useMutation({
    mutationFn: () =>
      backend
        .post<
          paths["/images/{id}/process/retry"]["post"]["responses"]["200"]["content"]["application/json"]
        >(`/images/${image.id}/process/retry`)
        .then((_) => _.data),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: [`/images/${image.id}/process`],
      });
      onSuccessSubmit && onSuccessSubmit();
    },
  });

  return (
    <div className="space-y-3">
      <p className="overflow-auto text-xs">{error}</p>
      <Button
        label="Try Again!"
        isLoading={isPending}
        onClick={() => retryLatestProcess()}
      />
    </div>
  );
}
