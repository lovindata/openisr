import { Button } from "@/v2/features/shared/components/molecules/Button";
import { useBackend } from "@/v2/services/backend";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface Props {
  error: string;
  imageId: number;
  onSuccessSubmit?: () => void;
}

export function ErrorContents({ error, imageId, onSuccessSubmit }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: retryLatestProcess, isPending } = useMutation({
    mutationFn: () =>
      backend
        .post(`/command/v1/images/${imageId}/process/retry`)
        .then(() => {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/query/v1/app/cards`] });
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
